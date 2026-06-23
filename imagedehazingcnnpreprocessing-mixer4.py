import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, BatchNormalization, Dropout,
    LayerNormalization, Dense, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import exposure

# === 1. Preprocessing Class dengan Histogram Matching dan CLAHE ===
class DehazingPreprocessor:
    """Kelas untuk preprocessing citra dengan histogram matching dan CLAHE"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def apply_clahe(self, image):
        """Aplikasi CLAHE untuk meningkatkan kontras lokal"""
        # Convert to uint8 if needed
        if image.dtype == np.float32:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        if len(image_uint8.shape) == 3:
            # Convert BGR to LAB
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2LAB)
            # Apply CLAHE to L channel
            lab[:,:,0] = self.clahe.apply(lab[:,:,0])
            # Convert back to BGR
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = self.clahe.apply(image_uint8)
        
        # Convert back to original dtype
        if image.dtype == np.float32:
            return result.astype(np.float32) / 255.0
        else:
            return result
    
    def histogram_matching(self, source, reference):
        """Histogram matching untuk menyamakan distribusi brightness dan contrast"""
        # Convert to uint8 if needed
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
        
        # Convert back to original dtype
        if source.dtype == np.float32:
            return matched.astype(np.float32) / 255.0
        else:
            return matched.astype(np.uint8)
    
    def preprocess_image_pair(self, hazy_img, clear_img):
        """Preprocessing lengkap untuk pasangan citra hazy dan clear"""
        # Histogram matching
        hazy_matched = self.histogram_matching(hazy_img, clear_img)
        
        # Apply CLAHE
        hazy_clahe = self.apply_clahe(hazy_matched)
        clear_clahe = self.apply_clahe(clear_img)
        
        return hazy_clahe, clear_clahe

# === 2. Load Dataset with Preprocessing and Augmentation ===
def load_dataset(clear_dir, hazy_dir, limit=None, use_preprocessing=True):
    X, y = [], []
    clear_files = {f.split('.')[0]: f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png'))}
    
    # Initialize preprocessor
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

            # Normalisasi ke float32
            clear_img = clear_img.astype(np.float32) / 255.0
            hazy_img = hazy_img.astype(np.float32) / 255.0

            # Apply preprocessing jika diaktifkan
            if use_preprocessing and preprocessor is not None:
                hazy_img, clear_img = preprocessor.preprocess_image_pair(hazy_img, clear_img)

            # Tambahkan original
            X.append(hazy_img)
            y.append(clear_img)
            
            # Tambahkan versi flip untuk augmentasi
            X.append(np.fliplr(hazy_img))
            y.append(np.fliplr(clear_img))
            
            # Tambahkan rotasi 90 derajat
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

# === MLP-Mixer components (Keras layer) ===
class MixerBlock(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim
        self.dropout_rate = dropout_rate

        # LayerNorms
        self.ln1 = LayerNormalization(axis=-1)
        self.ln2 = LayerNormalization(axis=-1)

        # Token-mixing: we will transpose (B, N, C) -> (B, C, N) and apply Dense on last axis (N)
        self.token_dense1 = Dense(self.token_mlp_dim)
        self.token_dense2 = Dense(self.num_patches)

        # Channel-mixing: standard FeedForward on last axis C
        self.channel_dense1 = Dense(self.channel_mlp_dim)
        self.channel_dense2 = Dense(self.embed_dim)

        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        # inputs: (B, N, C)
        # Token-mixing
        y = self.ln1(inputs)
        y = tf.transpose(y, perm=[0, 2, 1])          # (B, C, N)
        y = self.token_dense1(y)
        y = tf.nn.gelu(y)
        y = self.dropout(y, training=training)
        y = self.token_dense2(y)                     # (B, C, N)
        y = tf.transpose(y, perm=[0, 2, 1])          # (B, N, C)
        x = inputs + y

        # Channel-mixing
        z = self.ln2(x)
        z = self.channel_dense1(z)
        z = tf.nn.gelu(z)
        z = self.dropout(z, training=training)
        z = self.channel_dense2(z)
        return x + z

def mlp_mixer_on_feature_map(feature_map, patch_size=2, depth=2, token_mlp_dim=128, channel_mlp_dim=512, dropout_rate=0.0, name="mlp_mixer"):
    """
    feature_map: tensor (B, H, W, C)
    patch_size: int, spatial patch size to reduce tokens (helps performance)
    depth: number of MixerBlocks
    returns: tensor same spatial size as feature_map (B, H, W, C)
    """
    with tf.name_scope(name):
        # 1) Patch embedding: conv with kernel=patch_size, stride=patch_size
        #    produces (B, H', W', embed_dim) where embed_dim = C_in (preserve channels)
        C_in = int(feature_map.shape[-1])
        patch_embed = Conv2D(filters=C_in, kernel_size=patch_size, strides=patch_size, padding='same')(feature_map)
        # Get dynamic shape info
        def reshape_to_tokens(x):
            # x: (B, H', W', C)
            s = tf.shape(x)
            B = s[0]
            Hp = s[1]
            Wp = s[2]
            C = s[3]
            N = Hp * Wp
            # reshape to (B, N, C)
            x_resh = tf.reshape(x, (B, N, C))
            return x_resh

        tokens = Lambda(reshape_to_tokens, name=name + "_to_tokens")(patch_embed)  # (B, N, C)
        # num_patches (N) - dynamic: get via tf.shape when building MixerBlocks
        # But MixerBlock requires num_patches at construction. We can get static N if available; else use  - compute from patch_embed shape
        # Try to get static value if possible:
        Hp = patch_embed.shape[1]
        Wp = patch_embed.shape[2]
        if Hp is None or Wp is None:
            # fallback: compute N at runtime inside MixerBlock via token length: use tf.shape(tokens)[1]
            # To make MixerBlock construction easier, compute N runtime and pass a reasonably large value
            # However Dense layers need fixed output dim; we can still build Dense with units set from runtime using tokens' shape unknown -> to avoid complexity, compute N at graph build if possible.
            # For safety, compute N dynamically here:
            N_dynamic = tf.shape(tokens)[1]
            # convert to int by forcing to numpy if possible; else try to cast to int( ) - if fails, set a default (this happens rarely)
            try:
                num_patches = int(Hp) * int(Wp)
            except Exception:
                num_patches = None
        else:
            num_patches = Hp * Wp

        # If static num_patches known -> use normal MixerBlocks, else try to infer from tokens.shape
        if num_patches is None:
            # get dynamic N for Dense layer sizes - but Dense units must be integers at build time.
            # As a practical compromise: compute static num_patches from input size if available (should be available in our pipeline)
            # In most practical uses (fixed input sizes), patch_embed.shape is known. If not known, raise error to prompt user to use fixed input shapes.
            raise ValueError("mlp_mixer_on_feature_map requires known spatial dimensions at model build time. Use fixed input image size.")
        else:
            N = int(num_patches)

        # Create several MixerBlocks
        x_tokens = tokens
        for i in range(depth):
            x_tokens = MixerBlock(num_patches=N, embed_dim=C_in,
                                  token_mlp_dim=token_mlp_dim,
                                  channel_mlp_dim=channel_mlp_dim,
                                  dropout_rate=dropout_rate,
                                  name=f"{name}_block{i}")(x_tokens)

        # reshape tokens back to (B, H', W', C)
        def tokens_to_map(t):
            s_t = tf.shape(t)
            B = s_t[0]
            N = s_t[1]
            C = s_t[2]
            Hp_ = tf.shape(patch_embed)[1]
            Wp_ = tf.shape(patch_embed)[2]
            return tf.reshape(t, (B, Hp_, Wp_, C))
        map_back = Lambda(tokens_to_map, name=name + "_to_map")(x_tokens)

        # 2) Upsample back to original feature_map spatial size using Conv2DTranspose with strides=patch_size
        up = Conv2DTranspose(filters=C_in, kernel_size=patch_size, strides=patch_size, padding='same')(map_back)

        # Return upsampled feature map; caller may add residual
        return up

# === 3. Build Improved CNN Model (with inserted MLP-Mixer in bottleneck) ===
def build_model(mixer_patch_size=2, mixer_depth=2, mixer_token_mlp_dim=128, mixer_channel_mlp_dim=512, mixer_dropout=0.0):
    inputs = Input((256, 256, 3))

    # Encoder dengan lebih banyak layer
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

    # === INSERT MLP-MIXER HERE ===
    # Use small patch_size (2) to reduce token count: tokens = (H/patch_size)*(W/patch_size)
    try:
        x4_mixed = mlp_mixer_on_feature_map(
            x4,
            patch_size=mixer_patch_size,
            depth=mixer_depth,
            token_mlp_dim=mixer_token_mlp_dim,
            channel_mlp_dim=mixer_channel_mlp_dim,
            dropout_rate=mixer_dropout,
            name="bottleneck_mixer"
        )
        # Add residual to preserve CNN features
        x4 = Add()([x4, x4_mixed])
    except Exception as e:
        # If mixing fails due to unknown spatial dims, warn and continue with original x4
        tf.print("⚠️ Warning: MLP-Mixer insertion failed:", e)
        tf.print("⚠️ Continuing with original bottleneck (no mixer).")
        # x4 remains unchanged

    # Decoder dengan skip connections
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
    """Visualisasi perbandingan sebelum dan sesudah preprocessing"""
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
            
            # Normalisasi
            clear_img_norm = clear_img.astype(np.float32) / 255.0
            hazy_img_norm = hazy_img.astype(np.float32) / 255.0
            
            # Apply preprocessing
            hazy_processed, clear_processed = preprocessor.preprocess_image_pair(hazy_img_norm, clear_img_norm)
            
            # Show images
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

        if i < 5:  # Show first 5 results
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
    MODEL_SAVE_PATH = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\DATASET REVISI\MODEL REVISI\cnndehazing_SOTS8_MIX200_preprocessed_mixer.h5"

    print("=== Pure CNN Dehazing dengan Advanced Preprocessing + MLP-Mixer di Bottleneck ===")
    print("🔧 Fitur preprocessing: Histogram Matching + CLAHE")
    print("🚀 Augmentasi: Horizontal flip + Rotation")
    
    # Show preprocessing comparison
    print("\n1. Menampilkan perbandingan preprocessing...")
    show_preprocessing_comparison(CLEAR_DIR, HAZY_DIR, num_samples=2)

    # Load dataset with preprocessing
    print("\n2. Loading dataset dengan preprocessing...")
    X, y = load_dataset(CLEAR_DIR, HAZY_DIR, limit=200, use_preprocessing=True)
    
    print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"🔄 Train: {len(X_train)}, Validation: {len(X_val)}")

    # Build and compile model (you can tune mixer params here)
    print("\n3. 📦 Membuat model baru (dengan MLP-Mixer di bottleneck)...")
    model = build_model(
        mixer_patch_size=2,      # coba 2 atau 4; trade-off tokens vs compute
        mixer_depth=2,           # depth kecil untuk eksperimen ringan
        mixer_token_mlp_dim=128,
        mixer_channel_mlp_dim=512,
        mixer_dropout=0.1
    )
    print(f"📋 Model parameters: {model.count_params():,}")
    
    # Training
    print("\n4. 🚀 Training model dimulai...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ],
        verbose=1
    )

    # Save model
    print(f"\n5. 💾 Menyimpan model ke: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model berhasil disimpan!")

    # Evaluasi dan visualisasi
    print("\n6. 📊 Evaluasi dan visualisasi hasil...")
    show_sample_result(model, X_val, y_val, num_samples=3)
    
    avg_psnr, avg_ssim = evaluate_model(model, X_val, y_val, max_samples=20)
    
    plot_loss(history)
    
    print(f"\n🎉 Training selesai!")
    print(f"📈 Final PSNR: {avg_psnr:.2f} dB")
    print(f"📊 Final SSIM: {avg_ssim:.4f}")
    print(f"💾 Model tersimpan: {MODEL_SAVE_PATH}")
