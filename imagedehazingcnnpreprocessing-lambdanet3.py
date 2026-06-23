import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import exposure

# tambahan import tensorflow untuk LambdaLayer
import tensorflow as tf

# === LambdaLayer (LambdaNetworks) ===
class LambdaLayer(tf.keras.layers.Layer):
    """
    Simplified Lambda Layer suitable for CPU:
      - content lambda (global aggregation)
      - optional positional lambda approximated by Conv2D on V
    """
    def __init__(self,
                 dim_out,
                 n_heads=4,
                 dim_per_head=16,
                 use_positional=True,
                 pos_kernel=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_out = dim_out
        self.n_heads = n_heads
        self.dim_per_head = dim_per_head
        self.use_positional = use_positional
        self.pos_kernel = pos_kernel
        self.total_internal = n_heads * dim_per_head

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        # 1x1 conv projections
        self.q_proj = Conv2D(self.total_internal, kernel_size=1, padding='same', use_bias=False)
        self.k_proj = Conv2D(self.total_internal, kernel_size=1, padding='same', use_bias=False)
        self.v_proj = Conv2D(self.total_internal, kernel_size=1, padding='same', use_bias=False)
        if self.use_positional:
            self.pos_conv = Conv2D(self.total_internal, kernel_size=self.pos_kernel, padding='same', use_bias=False)
        self.out_proj = Conv2D(self.dim_out, kernel_size=1, padding='same', use_bias=False)
        super().build(input_shape)

    def call(self, x):
        # x: (B, H, W, C)
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        n = h * w

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to (B, N, heads, D)
        q = tf.reshape(q, (b, n, self.n_heads, self.dim_per_head))
        k = tf.reshape(k, (b, n, self.n_heads, self.dim_per_head))
        v = tf.reshape(v, (b, n, self.n_heads, self.dim_per_head))

        # content lambda
        k_soft = tf.nn.softmax(k, axis=1)  # normalize over positions
        lambda_c = tf.einsum('bnhk,bnhv->bhkv', k_soft, v)  # (B, heads, D, D)
        y_content = tf.einsum('bnhk,bhkv->bnhv', q, lambda_c)  # (B, N, heads, D)

        y = y_content

        # positional approx via conv on V
        if self.use_positional:
            v_spatial = tf.reshape(v, (b, h, w, self.total_internal))
            lambda_p = self.pos_conv(v_spatial)  # (B,H,W,heads*D)
            lambda_p = tf.reshape(lambda_p, (b, n, self.n_heads, self.dim_per_head))
            y_pos = q * lambda_p
            y = y + y_pos

        # collapse heads and dims
        y = tf.reshape(y, (b, n, self.total_internal))
        y = tf.reshape(y, (b, h, w, self.total_internal))
        out = self.out_proj(y)  # (B,H,W,dim_out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "dim_out": self.dim_out,
            "n_heads": self.n_heads,
            "dim_per_head": self.dim_per_head,
            "use_positional": self.use_positional,
            "pos_kernel": self.pos_kernel,
        })
        return cfg

# === 1. Preprocessing Class dengan Histogram Matching dan CLAHE ===
class DehazingPreprocessor:
    """Kelas untuk preprocessing citra dengan histogram matching dan CLAHE"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def apply_clahe(self, image):
        """Aplikasi CLAHE untuk meningkatkan kontras lokal"""
        if image.dtype == np.float32:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        if len(image_uint8.shape) == 3:
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = self.clahe.apply(lab[:,:,0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = self.clahe.apply(image_uint8)
        
        if image.dtype == np.float32:
            return result.astype(np.float32) / 255.0
        else:
            return result
    
    def histogram_matching(self, source, reference):
        """Histogram matching untuk menyamakan distribusi brightness dan contrast"""
        if source.dtype == np.float32:
            source_uint8 = (source * 255).astype(np.uint8)
            reference_uint8 = (reference * 255).astype(np.uint8)
        else:
            source_uint8 = source.astype(np.uint8)
            reference_uint8 = reference.astype(np.uint8)
        
        matched = np.zeros_like(source_uint8)
        
        if len(source_uint8.shape) == 3:
            for channel in range(source_uint8.shape[2]):
                matched[:,:,channel] = exposure.match_histograms(
                    source_uint8[:,:,channel], 
                    reference_uint8[:,:,channel]
                )
        else:
            matched = exposure.match_histograms(source_uint8, reference_uint8)
        
        if source.dtype == np.float32:
            return matched.astype(np.float32) / 255.0
        else:
            return matched.astype(np.uint8)
    
    def preprocess_image_pair(self, hazy_img, clear_img):
        """Preprocessing lengkap untuk pasangan citra hazy dan clear"""
        hazy_matched = self.histogram_matching(hazy_img, clear_img)
        hazy_clahe = self.apply_clahe(hazy_matched)
        clear_clahe = self.apply_clahe(clear_img)
        return hazy_clahe, clear_clahe

# === 2. Load Dataset with Preprocessing and Augmentation ===
def load_dataset(clear_dir, hazy_dir, limit=None, use_preprocessing=True):
    X, y = [], []
    clear_files = {f.split('.')[0]: f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png'))}
    
    if use_preprocessing:
        preprocessor = DehazingPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
        print("🔧 Preprocessing diaktifkan: Histogram Matching + CLAHE")
    else:
        preprocessor = None
        print("⚠️  Preprocessing dinonaktifkan")

    count = 0
    for fname in os.listdir(hazy_dir):
        if not fname.endswith(('.jpg', '.png')):
            continue

        hazy_id = fname.split("_")[0]
        if hazy_id in clear_files:
            clear_img = cv2.imread(os.path.join(clear_dir, clear_files[hazy_id]))
            hazy_img = cv2.imread(os.path.join(hazy_dir, fname))

            if clear_img is None or hazy_img is None:
                continue

            clear_img = cv2.resize(clear_img, (256, 256))
            hazy_img = cv2.resize(hazy_img, (256, 256))

            clear_img = clear_img.astype(np.float32) / 255.0
            hazy_img = hazy_img.astype(np.float32) / 255.0

            if use_preprocessing and preprocessor is not None:
                hazy_img, clear_img = preprocessor.preprocess_image_pair(hazy_img, clear_img)

            X.append(hazy_img)
            y.append(clear_img)
            X.append(np.fliplr(hazy_img))
            y.append(np.fliplr(clear_img))
            X.append(np.rot90(hazy_img))
            y.append(np.rot90(clear_img))

            count += 1
            if use_preprocessing:
                print(f"[✔] Pair {count} (dengan preprocessing): {fname} ↔ {clear_files[hazy_id]}")
            else:
                print(f"[✔] Pair {count}: {fname} ↔ {clear_files[hazy_id]}")
                
            if limit and count >= limit:
                break

    print(f"\n📊 Total data setelah augmentasi: {len(X)} pasang gambar")
    return np.array(X), np.array(y)

# === 3. Build Improved CNN Model (dengan LambdaLayer di bottleneck) ===
def build_model():
    inputs = Input((256, 256, 3))

    # Encoder
    x1 = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.1)(x1)
    x1p = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x1p)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2p = MaxPooling2D((2, 2))(x2)

    x3 = Conv2D(128, (3, 3), padding='same', activation='relu')(x2p)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(128, (3, 3), padding='same', activation='relu')(x3)
    x3p = MaxPooling2D((2, 2))(x3)

    # Bottleneck
    x4 = Conv2D(256, (3, 3), padding='same', activation='relu')(x3p)
    x4 = BatchNormalization()(x4)
    x4 = Conv2D(256, (3, 3), padding='same', activation='relu')(x4)

    # --- Insert LambdaLayer here to capture global context ---
    # CPU-friendly defaults: n_heads=4, dim_per_head=16 -> internal 64 channels
    x4 = LambdaLayer(dim_out=256, n_heads=4, dim_per_head=16, use_positional=True, pos_kernel=3)(x4)
    x4 = BatchNormalization()(x4)

    # Decoder dengan skip connections (struktur tetap seperti Anda minta)
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x4)
    x = Add()([x, x3])
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = Add()([x, x2])
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = Add()([x, x1])
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=MeanAbsoluteError(), metrics=['mse'])
    return model

# === 4. Visualisasi Preprocessing Results ===
def show_preprocessing_comparison(clear_dir, hazy_dir, num_samples=2):
    clear_files = {f.split('.')[0]: f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png'))}
    preprocessor = DehazingPreprocessor()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    count = 0
    for fname in os.listdir(hazy_dir):
        if count >= num_samples:
            break
            
        if not fname.endswith(('.jpg', '.png')):
            continue
            
        hazy_id = fname.split("_")[0]
        if hazy_id in clear_files:
            clear_img = cv2.imread(os.path.join(clear_dir, clear_files[hazy_id]))
            hazy_img = cv2.imread(os.path.join(hazy_dir, fname))
            
            if clear_img is None or hazy_img is None:
                continue
                
            clear_img = cv2.resize(clear_img, (256, 256))
            hazy_img = cv2.resize(hazy_img, (256, 256))
            
            clear_img_norm = clear_img.astype(np.float32) / 255.0
            hazy_img_norm = hazy_img.astype(np.float32) / 255.0
            
            hazy_processed, clear_processed = preprocessor.preprocess_image_pair(hazy_img_norm, clear_img_norm)
            
            axes[count, 0].imshow(cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB))
            axes[count, 0].set_title('Hazy Original')
            axes[count, 0].axis('off')
            
            axes[count, 1].imshow(cv2.cvtColor((hazy_processed * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[count, 1].set_title('Hazy Preprocessed')
            axes[count, 1].axis('off')
            
            axes[count, 2].imshow(cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB))
            axes[count, 2].set_title('Clear Original')
            axes[count, 2].axis('off')
            
            axes[count, 3].imshow(cv2.cvtColor((clear_processed * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[count, 3].set_title('Clear Preprocessed')
            axes[count, 3].axis('off')
            
            count += 1
    
    plt.tight_layout()
    plt.show()

# === 5. Visualisasi Hasil Sample ===
def show_sample_result(model, X_val, y_val, num_samples=3):
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axs = axs.reshape(1, -1)
        
    for i in range(min(num_samples, len(X_val))):
        pred = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)[0]

        axs[i][0].imshow(cv2.cvtColor((X_val[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[i][0].set_title("Hazy Input")
        axs[i][1].imshow(cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[i][1].set_title("Predicted")
        axs[i][2].imshow(cv2.cvtColor((y_val[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[i][2].set_title("Ground Truth")

        for ax in axs[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# === 6. Evaluasi PSNR dan SSIM ===
def evaluate_model(model, X_val, y_val, max_samples=10):
    psnr_list, ssim_list = [], []
    max_eval = min(len(X_val), max_samples)
    
    print(f"🔍 Evaluating {max_eval} samples...")
    
    for i in range(max_eval):
        pred = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)[0]
        gt = y_val[i]

        psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim_val = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        if i < 5:
            print(f"🖼️ Gambar ke-{i+1}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")

    print(f"\n📈 PSNR rata-rata: {np.mean(psnr_list):.2f} dB")
    print(f"📊 SSIM rata-rata: {np.mean(ssim_list):.4f}")
    return np.mean(psnr_list), np.mean(ssim_list)

# === 7. Visualisasi Loss ===
def plot_loss(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Training vs Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# === 8. Main Program ===
if __name__ == "__main__":
    CLEAR_DIR = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\SOTS\MIX\GT"
    HAZY_DIR  = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\SOTS\MIX\HAZY"
    MODEL_SAVE_PATH = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\DATASET REVISI\MODEL REVISI\cnndehazing_SOTS8_MIX200_lambda.h5"

    print("=== Pure CNN Dehazing dengan LambdaLayer (global context) ===")
    print("🔧 Fitur preprocessing: Histogram Matching + CLAHE")
    print("🚀 Augmentasi: Horizontal flip + Rotation")
    
    print("\n1. Menampilkan perbandingan preprocessing...")
    show_preprocessing_comparison(CLEAR_DIR, HAZY_DIR, num_samples=2)

    print("\n2. Loading dataset dengan preprocessing...")
    X, y = load_dataset(CLEAR_DIR, HAZY_DIR, limit=200, use_preprocessing=True)
    
    print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"🔄 Train: {len(X_train)}, Validation: {len(X_val)}")

    print("\n3. 📦 Membuat model baru...")
    model = build_model()
    print(f"📋 Model parameters: {model.count_params():,}")
    
    print("\n4. 🚀 Training model dimulai...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ],
        verbose=1
    )

    print(f"\n5. 💾 Menyimpan model ke: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model berhasil disimpan!")

    print("\n6. 📊 Evaluasi dan visualisasi hasil...")
    show_sample_result(model, X_val, y_val, num_samples=3)
    
    avg_psnr, avg_ssim = evaluate_model(model, X_val, y_val, max_samples=20)
    
    plot_loss(history)
    
    print(f"\n🎉 Training selesai!")
    print(f"📈 Final PSNR: {avg_psnr:.2f} dB")
    print(f"📊 Final SSIM: {avg_ssim:.4f}")
    print(f"💾 Model tersimpan: {MODEL_SAVE_PATH}")
