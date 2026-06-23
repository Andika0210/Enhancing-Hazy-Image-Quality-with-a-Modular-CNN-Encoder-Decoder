"""
Dehazing CNN + MLP-Mixer (light skip, non U-Net) — FIXED
- Konsistensi token Mixer: PATCH_SIZE diterapkan pada feature-map encoder (32x32).
- num_patches = gh * gw (bukan dari input).
- Default PATCH_SIZE=8 -> 4x4=16 token (stabil). Bisa pakai 4/8/16 selama 32 % PATCH_SIZE == 0.
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D,
    BatchNormalization, Add, LayerNormalization,
    Dense, Reshape, UpSampling2D, Dropout
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ----------------------------------------------------------------------------- #
# Konfigurasi
# ----------------------------------------------------------------------------- #
CONFIG = {
    "INPUT_SHAPE": (128, 128, 3),
    # PATCH_SIZE di sini adalah ukuran patch pada feature-map encoder (bukan pada input).
    # Dengan input 128 dan 2x MaxPool (→ 32x32), maka PATCH_SIZE harus membagi 32 (4/8/16).
    "PATCH_SIZE": 8,
    "EMBED_DIM": 64,
    "NUM_BLOCKS": 3,
    "TOKEN_MLP_DIM": 128,
    "CHANNEL_MLP_DIM": 256,
    "BATCH_SIZE": 8,
    "EPOCHS": 40,
    "DATA_LIMIT": None,
    "LR": 1e-4,
    "USE_PERCEPTUAL": False,
    "PERCEPTUAL_WEIGHT": 0.1,
    "ALPHA_L1": 0.7,
    "VAL_SPLIT": 0.15,
    "SEED": 42,
}

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
tf.random.set_seed(CONFIG["SEED"])

# ----------------------------------------------------------------------------- #
# Smart Augmentation
# ----------------------------------------------------------------------------- #
class SmartAugmentation:
    def __init__(self, aug_prob=0.7, seed=None):
        self.aug_prob = aug_prob
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

    def dark_channel_prior(self, img, patch_size=15):
        min_ch = np.min(img, axis=2)
        min_u8 = (np.clip(min_ch, 0, 1) * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_u8 = cv2.erode(min_u8, kernel)
        return dark_u8.astype(np.float32) / 255.0

    def simulate_haze_variation(self, clear_img):
        if random.random() > self.aug_prob:
            return clear_img.astype(np.float32)
        clear = clear_img.astype(np.float32)
        dark = self.dark_channel_prior(clear, patch_size=15)
        A = np.random.uniform(0.7, 1.0, 3).astype(np.float32)
        beta = np.random.uniform(0.6, 2.0)
        t = np.exp(-beta * dark).astype(np.float32)
        t = np.clip(t, 0.05, 1.0)
        hazy = np.empty_like(clear, dtype=np.float32)
        for c in range(3):
            hazy[:, :, c] = clear[:, :, c] * t + A[c] * (1.0 - t)
        return np.clip(hazy, 0.0, 1.0)

    def photometric_augmentation(self, img):
        if random.random() > self.aug_prob:
            return img.astype(np.float32)
        aug = img.astype(np.float32)
        if random.random() < 0.5:  # brightness
            b = np.random.uniform(-0.12, 0.12)
            aug = np.clip(aug + b, 0, 1)
        if random.random() < 0.5:  # contrast
            c = np.random.uniform(0.85, 1.15)
            aug = np.clip((aug - 0.5) * c + 0.5, 0, 1)
        if random.random() < 0.5:  # gamma
            g = np.random.uniform(0.8, 1.25)
            aug = np.power(np.clip(aug, 1e-6, 1.0), g)
        if random.random() < 0.3:  # channel shift
            shift = np.random.uniform(-0.04, 0.04, 3).reshape((1, 1, 3))
            aug = np.clip(aug + shift, 0, 1)
        return aug

    def geometric_augmentation(self, img):
        if random.random() > self.aug_prob:
            return img.astype(np.float32)
        aug = img.astype(np.float32); h, w = aug.shape[:2]
        if random.random() < 0.4:
            angle = np.random.uniform(-12, 12)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug = cv2.warpAffine((aug*255).astype(np.uint8), M, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            aug = aug.astype(np.float32) / 255.0
        if random.random() < 0.5: aug = np.fliplr(aug)
        if random.random() < 0.2: aug = np.flipud(aug)
        if random.random() < 0.3:
            crop = int(min(h, w) * np.random.uniform(0.82, 0.95))
            sx = np.random.randint(0, w - crop + 1); sy = np.random.randint(0, h - crop + 1)
            aug = aug[sy:sy+crop, sx:sx+crop]
            aug = cv2.resize((aug*255).astype(np.uint8), (w, h)).astype(np.float32)/255.0
        return np.clip(aug, 0.0, 1.0)

    def apply_all_augmentations(self, clear_img, hazy_img):
        clear = clear_img.astype(np.float32)
        hazy  = hazy_img.astype(np.float32)
        # geometric sinkron
        if random.random() < 0.6:
            rs = random.getstate(); np_rs = np.random.get_state()
            clear_g = self.geometric_augmentation(clear)
            random.setstate(rs); np.random.set_state(np_rs)
            hazy_g  = self.geometric_augmentation(hazy)
        else:
            clear_g = clear; hazy_g = hazy
        # photometric independen
        clear_p = self.photometric_augmentation(clear_g)
        hazy_p  = self.photometric_augmentation(hazy_g)
        # tambahan haze sintetis 30%
        if random.random() < 0.3:
            hazy_s = self.simulate_haze_variation(clear_p)
            return clear_p, hazy_s
        return clear_p, hazy_p

# ----------------------------------------------------------------------------- #
# Dataset utilities
# ----------------------------------------------------------------------------- #
def load_dataset(clear_dir, hazy_dir, input_shape, limit=None, verbose=True):
    H, W, _ = input_shape
    X, y = [], []
    # peta nama GT
    clear_files = {}
    for f in os.listdir(clear_dir):
        if not f.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        clear_files[os.path.splitext(f)[0]] = f

    def stems(fname):
        s = os.path.splitext(fname)[0]; out = [s]
        if '_' in s: out.append(s.split('_')[0])
        if '-' in s: out.append(s.split('-')[0])
        if '.' in s: out.append(s.split('.')[0])
        return out

    cnt = 0
    for fname in os.listdir(hazy_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        found = None
        for s in stems(fname):
            if s in clear_files:
                found = clear_files[s]; break
        if found is None:
            if verbose: print(f"[!] No GT for {fname} (skip)")
            continue

        # BGR->RGB agar konsisten
        c_bgr = cv2.imread(os.path.join(clear_dir, found))
        h_bgr = cv2.imread(os.path.join(hazy_dir, fname))
        if c_bgr is None or h_bgr is None: continue
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
        h_rgb = cv2.cvtColor(h_bgr, cv2.COLOR_BGR2RGB)

        cimg = cv2.resize(c_rgb, (W, H)).astype(np.float32) / 255.0
        himg = cv2.resize(h_rgb, (W, H)).astype(np.float32) / 255.0
        X.append(himg); y.append(cimg)
        cnt += 1
        if verbose: print(f"[✔] Pair {cnt}: {fname} ↔ {found}")
        if limit and cnt >= limit: break

    return np.array(X, np.float32), np.array(y, np.float32)

class AugmentedDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=8, augment=True, input_shape=(128,128,3)):
        self.X = X; self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.augmentor = SmartAugmentation(seed=CONFIG["SEED"]) if augment else None
        self.indices = np.arange(len(X))
        self.input_shape = input_shape

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        H,W,C = self.input_shape
        bx = np.zeros((len(inds),H,W,C), np.float32)
        by = np.zeros((len(inds),H,W,C), np.float32)
        for i,j in enumerate(inds):
            clear = self.y[j].copy()
            hazy  = self.X[j].copy()
            if self.augment and self.augmentor:
                clear, hazy = self.augmentor.apply_all_augmentations(clear, hazy)
            bx[i] = hazy; by[i] = clear
        return bx, by

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ----------------------------------------------------------------------------- #
# Loss: MAE + (1 - SSIM) [+ optional Perceptual]
# ----------------------------------------------------------------------------- #
def mae_ssim_loss(alpha=0.7, data_range=1.0):
    mae = tf.keras.losses.MeanAbsoluteError()
    def loss(y_true, y_pred):
        l1 = mae(y_true, y_pred)
        ssim = tf.image.ssim(y_true, y_pred, max_val=data_range)
        return alpha * l1 + (1.0 - alpha) * (1.0 - tf.reduce_mean(ssim))
    return loss

def perceptual_loss_factory(weight=0.1):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(128,128,3))
    vgg.trainable = False
    feat = Model(vgg.input, [vgg.get_layer('block3_conv1').output,
                             vgg.get_layer('block4_conv1').output])
    mae = tf.keras.losses.MeanAbsoluteError()
    def preprocess(img):
        img = tf.image.resize(img, (128,128))
        return tf.keras.applications.vgg19.preprocess_input(img*255.0)
    def ploss(y_true, y_pred):
        yt, yp = preprocess(y_true), preprocess(y_pred)
        ft, fp = feat(yt), feat(yp)
        return weight * sum(mae(a,b) for a,b in zip(ft, fp))
    return ploss

def total_loss_builder(use_perceptual=False, alpha=0.7, perceptual_weight=0.1):
    base = mae_ssim_loss(alpha=alpha, data_range=1.0)
    if not use_perceptual:
        return base
    p = perceptual_loss_factory(weight=perceptual_weight)
    def total(y_true, y_pred):
        return base(y_true, y_pred) + p(y_true, y_pred)
    return total

# ----------------------------------------------------------------------------- #
# MLP-Mixer block
# ----------------------------------------------------------------------------- #
class MixerBlock(layers.Layer):
    def __init__(self, num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout=0.0, **kw):
        super().__init__(**kw)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.t1  = Dense(token_mlp_dim, activation='gelu')
        self.t2  = Dense(num_patches)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.c1  = Dense(channel_mlp_dim, activation='gelu')
        self.c2  = Dense(embed_dim)
        self.do  = Dropout(dropout)

    def call(self, x, training=False):
        # Token mixing
        y = self.ln1(x)
        y = tf.transpose(y, [0,2,1])        # (B, C, T)
        y = self.t1(y)
        y = self.do(y, training=training)
        y = self.t2(y)
        y = tf.transpose(y, [0,2,1])        # (B, T, C)
        x = x + y
        # Channel mixing
        z = self.ln2(x)
        z = self.c1(z)
        z = self.do(z, training=training)
        z = self.c2(z)
        return x + z

# ----------------------------------------------------------------------------- #
# Model: CNN encoder -> Mixer bottleneck -> decoder ringan + light skip
# ----------------------------------------------------------------------------- #
def build_cnn_mlp_mixer(input_shape=(128,128,3),
                        patch_size=8,
                        embed_dim=64,
                        num_blocks=3,
                        token_mlp_dim=128,
                        channel_mlp_dim=256,
                        dropout=0.1):
    H, W, C = input_shape
    assert (H % 4 == 0) and (W % 4 == 0), "Dengan 2x pooling, H/W harus kelipatan 4"
    enc_h = H // 4  # 32
    enc_w = W // 4  # 32
    assert (enc_h % patch_size == 0) and (enc_w % patch_size == 0), \
        f"PATCH_SIZE harus membagi {enc_h} (pakai 4/8/16)."

    inputs = Input(shape=input_shape)

    # Encoder (2x pooling → 32x32), simpan skip
    c1 = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D(2)(c1)  # 64x64

    c2 = Conv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D(2)(c2)  # 32x32

    c3 = Conv2D(embed_dim, 3, padding='same', activation='relu')(p2)
    c3 = BatchNormalization()(c3)  # 32x32xEMBED

    # Patch embedding pada feature 32x32 (BUKAN pada input)
    patch_embed = Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(c3)
    gh = enc_h // patch_size
    gw = enc_w // patch_size
    num_patches = gh * gw
    x = Reshape((num_patches, embed_dim))(patch_embed)

    # Mixer blocks
    for _ in range(num_blocks):
        x = MixerBlock(num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout=dropout)(x)

    # Balik ke grid gh x gw lalu upsample ke 32x32
    x = Reshape((gh, gw, embed_dim))(x)
    if patch_size > 1:  # upsample kembali ke 32x32
        x = UpSampling2D(size=(patch_size, patch_size), interpolation='bilinear')(x)
        x = Conv2D(embed_dim, 3, padding='same', activation='relu')(x)

    # Decoder ringan + light skip (bukan U-Net penuh)
    d1 = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)   # 32->64
    d1 = Add()([d1, c2])
    d1 = Conv2D(64, 3, padding='same', activation='relu')(d1)

    d2 = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(d1)  # 64->128
    d2 = Add()([d2, c1])
    d2 = Conv2D(32, 3, padding='same', activation='relu')(d2)

    out = Conv2D(16, 3, padding='same', activation='relu')(d2)
    outputs = Conv2D(3, 1, activation='sigmoid')(out)

    return Model(inputs, outputs, name="cnn_mlp_mixer_lightskip_fix")

# ----------------------------------------------------------------------------- #
# Visualisasi & Evaluasi
# ----------------------------------------------------------------------------- #
def show_sample_result(model, X_val, y_val, n=3):
    n = min(n, len(X_val))
    fig, axs = plt.subplots(n, 3, figsize=(10, 3*n))
    for i in range(n):
        pred = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)[0]
        axs[i,0].imshow(np.clip(X_val[i],0,1)); axs[i,0].set_title("Hazy"); axs[i,0].axis('off')
        axs[i,1].imshow(np.clip(pred,0,1));     axs[i,1].set_title("Pred"); axs[i,1].axis('off')
        axs[i,2].imshow(np.clip(y_val[i],0,1)); axs[i,2].set_title("GT");   axs[i,2].axis('off')
    plt.tight_layout(); plt.show()

def evaluate_model(model, X_val, y_val, max_samples=50):
    n = min(len(X_val), max_samples)
    psnr_list, ssim_list = [], []
    for i in range(n):
        pred = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)[0]
        gt = y_val[i]
        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)
        psnr_list.append(psnr); ssim_list.append(ssim)
        print(f"[{i+1}/{n}] PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    print(f"\nAvg PSNR: {np.mean(psnr_list):.2f} dB | Avg SSIM: {np.mean(ssim_list):.4f}")

# ----------------------------------------------------------------------------- #
# Inference folder (opsional)
# ----------------------------------------------------------------------------- #
def match_gt_for_file(hazy_fname, gt_dir):
    stem = os.path.splitext(os.path.basename(hazy_fname))[0]
    cands = [stem]
    if '_' in stem: cands.append(stem.split('_')[0])
    if '-' in stem: cands.append(stem.split('-')[0])
    if '.' in stem: cands.append(stem.split('.')[0])
    for c in cands:
        for ext in ('.png','.jpg','.jpeg','.bmp'):
            p = os.path.join(gt_dir, c + ext)
            if os.path.exists(p): return p
    return None

def prepare_image_for_model(img_path, input_shape):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = input_shape
    img_res = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    return img_res.astype(np.float32) / 255.0

def test_images_and_show(model_path, hazy_dir, gt_dir=None, output_dir=None,
                         input_shape=(128,128,3), max_images=None, show=True):
    co = {'MixerBlock': MixerBlock}
    model = load_model(model_path, custom_objects=co, compile=False)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    files.sort()
    if max_images: files = files[:max_images]

    results = []
    for fname in files:
        fpath = os.path.join(hazy_dir, fname)
        hazy = prepare_image_for_model(fpath, input_shape)
        pred = model.predict(np.expand_dims(hazy, 0), verbose=0)[0]
        pred = np.clip(pred, 0, 1)

        psnr_val = None; ssim_val = None; gt_img = None
        if gt_dir is not None:
            gt_path = match_gt_for_file(fname, gt_dir)
            if gt_path:
                gt_img = prepare_image_for_model(gt_path, input_shape)
                psnr_val = peak_signal_noise_ratio(gt_img, pred, data_range=1.0)
                ssim_val = structural_similarity(gt_img, pred, channel_axis=-1, data_range=1.0)

        if output_dir:
            out_u8 = (pred*255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, os.path.splitext(fname)[0] + "_dehazed.png"), out_bgr)

        results.append({"fname": fname, "psnr": psnr_val, "ssim": ssim_val})

        if show:
            cols = 3 if gt_img is not None else 2
            fig, axs = plt.subplots(1, cols, figsize=(4*cols, 4))
            axs = np.atleast_1d(axs)
            axs[0].imshow(hazy); axs[0].set_title("Hazy"); axs[0].axis('off')
            axs[1].imshow(pred); axs[1].set_title("Dehazed"); axs[1].axis('off')
            if gt_img is not None:
                axs[2].imshow(gt_img); axs[2].set_title("GT"); axs[2].axis('off')
                fig.suptitle(f"{fname} | PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")
            else:
                fig.suptitle(fname)
            plt.tight_layout(); plt.show()

    print("\n===== Inference Summary =====")
    for r in results:
        if r["psnr"] is not None:
            print(f"{r['fname']}: PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")
        else:
            print(f"{r['fname']}: (no GT)")

    return results

# ----------------------------------------------------------------------------- #
# Training script
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ==== GANTI PATH DI SINI ====
    CLEAR_DIR = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\SOTS\MIX\GT"
    HAZY_DIR  = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\SOTS\MIX\HAZY"
    SAVE_PATH = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\MODEL_REVISI\cnn_mlp_mixer_lightskip_fix.h5"
    # ============================

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("📁 Loading dataset ...")
    X, y = load_dataset(CLEAR_DIR, HAZY_DIR, input_shape=CONFIG["INPUT_SHAPE"], limit=CONFIG["DATA_LIMIT"])
    if len(X) == 0:
        raise RuntimeError("Dataset kosong. Cek path CLEAR_DIR & HAZY_DIR.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG["VAL_SPLIT"], random_state=CONFIG["SEED"]
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    print("🧠 Building model ...")
    model = build_cnn_mlp_mixer(
        input_shape=CONFIG["INPUT_SHAPE"],
        patch_size=CONFIG["PATCH_SIZE"],
        embed_dim=CONFIG["EMBED_DIM"],
        num_blocks=CONFIG["NUM_BLOCKS"],
        token_mlp_dim=CONFIG["TOKEN_MLP_DIM"],
        channel_mlp_dim=CONFIG["CHANNEL_MLP_DIM"],
        dropout=0.1
    )
    loss_fn = total_loss_builder(
        use_perceptual=CONFIG["USE_PERCEPTUAL"],
        alpha=CONFIG["ALPHA_L1"],
        perceptual_weight=CONFIG["PERCEPTUAL_WEIGHT"]
    )
    model.compile(optimizer=Adam(CONFIG["LR"]), loss=loss_fn, metrics=['mae'])
    model.summary()

    # Callbacks
    ckpt_path = SAVE_PATH.replace(".h5", "_best.h5")
    cbs = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ]

    print("🎛️ Preparing generators ...")
    train_gen = AugmentedDataGenerator(
        X_train, y_train, batch_size=CONFIG["BATCH_SIZE"], augment=True,  input_shape=CONFIG["INPUT_SHAPE"]
    )
    val_gen   = AugmentedDataGenerator(
        X_val,   y_val,   batch_size=CONFIG["BATCH_SIZE"], augment=False, input_shape=CONFIG["INPUT_SHAPE"]
    )

    print("🚀 Training ...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["EPOCHS"],
        callbacks=cbs,
        verbose=1
    )

    print(f"💾 Saving model to {SAVE_PATH}")
    model.save(SAVE_PATH)

    print("📊 Evaluasi (val split) ...")
    show_sample_result(model, X_val, y_val, n=3)
    evaluate_model(model, X_val, y_val, max_samples=50)
