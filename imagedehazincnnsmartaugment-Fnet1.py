import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, BatchNormalization, Dropout, Lambda, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import random
from scipy.ndimage import rotate
import tensorflow as tf

# === 1. FNet Layer Implementation ===
class FNetLayer(tf.keras.layers.Layer):
    """
    FNet layer yang menggunakan 2D Fourier Transform untuk token mixing
    Menggantikan self-attention dengan operasi FFT yang lebih efisien
    """
    def __init__(self, **kwargs):
        super(FNetLayer, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def build(self, input_shape):
        super(FNetLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs shape: (batch_size, height, width, channels)
        # Transpose untuk FFT: (batch_size, channels, height, width)
        transposed = tf.transpose(inputs, [0, 3, 1, 2])
        
        # Apply 2D FFT pada spatial dimensions untuk setiap channel
        fft_result = tf.signal.fft2d(tf.cast(transposed, tf.complex64))
        
        # Ambil bagian real dan imag, lalu gabungkan untuk mixing
        real_part = tf.math.real(fft_result)
        imag_part = tf.math.imag(fft_result)
        
        # Kombinasi real dan imag untuk feature mixing
        mixed_features = real_part + 0.1 * imag_part
        
        # Transpose kembali ke format asli
        mixed_features = tf.transpose(mixed_features, [0, 2, 3, 1])
        
        # Normalisasi untuk stabilitas
        normalized = self.layer_norm(mixed_features)
        
        return normalized
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(FNetLayer, self).get_config()
        return config

# === 2. FNet Block untuk Feature Mixing ===
def fnet_block(inputs, filters, dropout_rate=0.1, block_id=None):
    """
    Blok FNet yang menggabungkan Fourier mixing dengan CNN
    """
    # FNet mixing layer dengan unique name
    layer_name = f"fnet_mixing_{filters}" if block_id is None else f"fnet_mixing_{filters}_{block_id}"
    fnet_output = FNetLayer(name=layer_name)(inputs)
    
    # Residual connection dengan input (pastikan shape sama)
    if inputs.shape[-1] == fnet_output.shape[-1]:
        mixed = Add()([inputs, fnet_output])
    else:
        # Jika channel berbeda, gunakan projection
        proj_name = f"proj_{filters}_{block_id}" if block_id else f"proj_{filters}"
        projected_input = Conv2D(filters, (1, 1), padding='same', name=proj_name)(inputs)
        mixed = Add()([projected_input, fnet_output])
    
    mixed = BatchNormalization()(mixed)
    
    # Feed-forward network dengan CNN
    ff = Conv2D(filters * 2, (1, 1), activation='relu')(mixed)
    ff = Dropout(dropout_rate)(ff)
    ff = Conv2D(filters, (1, 1))(ff)
    
    # Residual connection
    output = Add()([mixed, ff])
    output = BatchNormalization()(output)
    
    return output

# === Alternative Simple FNet Block ===
def simple_fnet_block(inputs, filters, dropout_rate=0.1, block_id=None):
    """
    Versi sederhana FNet block untuk CPU yang terbatas
    """
    # Simplified Fourier mixing dengan unique names
    x = inputs
    
    # Apply FFT mixing secara sederhana dengan unique name
    layer_name = f"simple_fnet_{filters}" if block_id is None else f"simple_fnet_{filters}_{block_id}"
    fft_mixed = FNetLayer(name=layer_name)(x)
    
    # Simple residual dan normalization
    if x.shape[-1] == filters:
        mixed = Add()([x, fft_mixed])
    else:
        x_proj = Conv2D(filters, (1, 1), padding='same', name=f"proj_{filters}_{block_id}" if block_id else None)(x)
        mixed = Add()([x_proj, fft_mixed])
    
    output = BatchNormalization()(mixed)
    output = Dropout(dropout_rate)(output)
    
    return output

# === 3. Augmentasi Data Pintar (sama seperti sebelumnya) ===
class SmartAugmentation:
    def __init__(self):
        self.aug_prob = 0.7
    
    def dark_channel_prior(self, img, patch_size=15):
        min_channel = np.min(img, axis=2)
        kernel = np.ones((patch_size, patch_size), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def simulate_haze_variation(self, clear_img):
        if random.random() > self.aug_prob:
            return clear_img
            
        dark_ch = self.dark_channel_prior(clear_img)
        A = np.random.uniform(0.7, 1.0, 3)
        beta = np.random.uniform(0.5, 2.0)
        
        t = 1 - np.random.uniform(0.1, 0.7) * dark_ch
        t = np.clip(t, 0.1, 1.0)
        
        hazy_img = np.zeros_like(clear_img)
        for c in range(3):
            hazy_img[:,:,c] = clear_img[:,:,c] * t + A[c] * (1 - t)
        
        return np.clip(hazy_img, 0, 1)
    
    def photometric_augmentation(self, img):
        if random.random() > self.aug_prob:
            return img
            
        aug_img = img.copy()
        
        if random.random() < 0.5:
            brightness = np.random.uniform(-0.1, 0.1)
            aug_img = np.clip(aug_img + brightness, 0, 1)
        
        if random.random() < 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            aug_img = np.clip((aug_img - 0.5) * contrast + 0.5, 0, 1)
        
        if random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.3)
            aug_img = np.power(aug_img, gamma)
        
        if random.random() < 0.3:
            color_shift = np.random.uniform(-0.05, 0.05, 3)
            aug_img = aug_img + color_shift
            aug_img = np.clip(aug_img, 0, 1)
        
        return aug_img
    
    def geometric_augmentation(self, img):
        if random.random() > self.aug_prob:
            return img
            
        aug_img = img.copy()
        h, w = img.shape[:2]
        
        if random.random() < 0.4:
            angle = np.random.uniform(-15, 15)
            aug_img = rotate(aug_img, angle, reshape=False, mode='reflect')
        
        if random.random() < 0.5:
            aug_img = np.fliplr(aug_img)
        
        if random.random() < 0.2:
            aug_img = np.flipud(aug_img)
        
        if random.random() < 0.3:
            crop_size = int(min(h, w) * np.random.uniform(0.8, 0.95))
            start_x = np.random.randint(0, w - crop_size + 1)
            start_y = np.random.randint(0, h - crop_size + 1)
            
            cropped = aug_img[start_y:start_y+crop_size, start_x:start_x+crop_size]
            aug_img = cv2.resize(cropped, (w, h))
        
        return np.clip(aug_img, 0, 1)
    
    def apply_all_augmentations(self, clear_img, hazy_img):
        if random.random() < 0.6:
            random_state = random.getstate()
            np_random_state = np.random.get_state()
            
            aug_clear = self.geometric_augmentation(clear_img)
            
            random.setstate(random_state)
            np.random.set_state(np_random_state)
            aug_hazy = self.geometric_augmentation(hazy_img)
        else:
            aug_clear = clear_img.copy()
            aug_hazy = hazy_img.copy()
        
        aug_clear = self.photometric_augmentation(aug_clear)
        aug_hazy = self.photometric_augmentation(aug_hazy)
        
        if random.random() < 0.3:
            synthetic_hazy = self.simulate_haze_variation(aug_clear)
            return aug_clear, synthetic_hazy
        
        return aug_clear, aug_hazy

# === 4. Data Generator ===
class AugmentedDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=8, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.augmentor = SmartAugmentation() if augment else None
        self.indices = np.arange(len(X))
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = np.zeros((len(batch_indices), 256, 256, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_indices), 256, 256, 3), dtype=np.float32)
        
        for i, data_idx in enumerate(batch_indices):
            clear_img = self.y[data_idx].copy()
            hazy_img = self.X[data_idx].copy()
            
            if self.augment and self.augmentor:
                clear_img, hazy_img = self.augmentor.apply_all_augmentations(clear_img, hazy_img)
            
            batch_y[i] = clear_img
            batch_x[i] = hazy_img
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# === 5. Load Dataset ===
def load_dataset(clear_dir, hazy_dir, limit=None):
    X, y = [], []
    clear_files = {f.split('.')[0]: f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png'))}

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

            X.append(hazy_img)
            y.append(clear_img)

            count += 1
            print(f"[✔] Pair {count}: {fname} ↔ {clear_files[hazy_id]}")
            if limit and count >= limit:
                break

    return np.array(X), np.array(y)

# === 6. Build CNN-FNet Hybrid Model ===
def build_cnn_fnet_model(use_simple_fnet=True):
    """
    Model hybrid CNN-FNet untuk dehazing
    Menggunakan FNet untuk mixing features pada berbagai level
    """
    inputs = Input((256, 256, 3))
    
    # Pilih jenis FNet block
    fnet_fn = simple_fnet_block if use_simple_fnet else fnet_block

    # === ENCODER PATH ===
    # Level 1: Initial feature extraction
    x1 = Conv2D(32, (7, 7), padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (5, 5), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    
    # FNet mixing untuk level 1
    x1_fnet = fnet_fn(x1, 32, dropout_rate=0.1, block_id="level1")
    x1_fnet = Dropout(0.1)(x1_fnet)
    x1p = MaxPooling2D((2, 2))(x1_fnet)

    # Level 2: Middle feature extraction
    x2 = Conv2D(64, (5, 5), padding='same', activation='relu')(x1p)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    
    # FNet mixing untuk level 2
    x2_fnet = fnet_fn(x2, 64, dropout_rate=0.1, block_id="level2")
    x2_fnet = Dropout(0.1)(x2_fnet)
    x2p = MaxPooling2D((2, 2))(x2_fnet)

    # Level 3: Deep feature extraction
    x3 = Conv2D(128, (3, 3), padding='same', activation='relu')(x2p)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(128, (3, 3), padding='same', activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    
    # FNet mixing untuk level 3
    x3_fnet = fnet_fn(x3, 128, dropout_rate=0.15, block_id="level3")
    x3_fnet = Dropout(0.2)(x3_fnet)
    x3p = MaxPooling2D((2, 2))(x3_fnet)

    # === BOTTLENECK dengan FNet ===
    bottleneck = Conv2D(256, (3, 3), padding='same', activation='relu')(x3p)
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = Conv2D(256, (3, 3), padding='same', activation='relu')(bottleneck)
    bottleneck = BatchNormalization()(bottleneck)
    
    # FNet mixing di bottleneck
    bottleneck_fnet = fnet_fn(bottleneck, 256, dropout_rate=0.2, block_id="bottleneck")
    if not use_simple_fnet:  # Double FNet hanya untuk full version
        bottleneck_fnet = fnet_fn(bottleneck_fnet, 256, dropout_rate=0.2, block_id="bottleneck2")
    bottleneck_final = Dropout(0.3)(bottleneck_fnet)

    # === DECODER PATH ===
    # Level 3 decode
    up3 = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(bottleneck_final)
    up3 = Add()([up3, x3_fnet])  # Skip connection dengan FNet features
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    
    # FNet mixing di decoder
    up3_fnet = fnet_fn(up3, 128, dropout_rate=0.1, block_id="decode3")

    # Level 2 decode
    up2 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(up3_fnet)
    up2 = Add()([up2, x2_fnet])
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    
    # FNet mixing di decoder
    up2_fnet = fnet_fn(up2, 64, dropout_rate=0.1, block_id="decode2")

    # Level 1 decode
    up1 = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(up2_fnet)
    up1 = Add()([up1, x1_fnet])
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(32, (5, 5), padding='same', activation='relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)

    # Final refinement dengan FNet
    refined = fnet_fn(up1, 32, dropout_rate=0.05, block_id="final")
    refined = Conv2D(16, (3, 3), padding='same', activation='relu')(refined)
    refined = BatchNormalization()(refined)
    
    # Output layer
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(refined)

    model_name = 'CNN_SimpleFNet_Dehazing' if use_simple_fnet else 'CNN_FNet_Dehazing'
    model = Model(inputs, outputs, name=model_name)
    
    # Compile dengan learning rate yang lebih kecil untuk stabilitas
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=MeanAbsoluteError(), metrics=['mae'])
    
    return model

# === 7. Model CNN Murni untuk Perbandingan ===
def build_pure_cnn_model():
    """Model CNN murni tanpa FNet untuk perbandingan"""
    inputs = Input((256, 256, 3))

    # Encoder
    x1 = Conv2D(32, (7, 7), padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (5, 5), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.1)(x1)
    x1p = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(64, (5, 5), padding='same', activation='relu')(x1p)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.1)(x2)
    x2p = MaxPooling2D((2, 2))(x2)

    x3 = Conv2D(128, (3, 3), padding='same', activation='relu')(x2p)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(128, (3, 3), padding='same', activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.2)(x3)
    x3p = MaxPooling2D((2, 2))(x3)

    # Bottleneck
    bottleneck = Conv2D(256, (3, 3), padding='same', activation='relu')(x3p)
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = Conv2D(256, (3, 3), padding='same', activation='relu')(bottleneck)
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = Dropout(0.3)(bottleneck)

    # Decoder
    up3 = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(bottleneck)
    up3 = Add()([up3, x3])
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)

    up2 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(up3)
    up2 = Add()([up2, x2])
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)

    up1 = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(up2)
    up1 = Add()([up1, x1])
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(32, (5, 5), padding='same', activation='relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)

    refined = Conv2D(16, (3, 3), padding='same', activation='relu')(up1)
    refined = BatchNormalization()(refined)
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(refined)

    model = Model(inputs, outputs, name='Pure_CNN_Dehazing')
    model.compile(optimizer='adam', loss=MeanAbsoluteError(), metrics=['mae'])
    return model

# === 8. Visualisasi dan Evaluasi ===
def show_sample_result(model, X_val, y_val, model_name="Model"):
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Sample Results - {model_name}', fontsize=16)
    
    for i in range(3):
        pred = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)[0]

        axs[i][0].imshow(X_val[i])
        axs[i][0].set_title("Hazy Input")
        axs[i][1].imshow(pred)
        axs[i][1].set_title("Predicted")
        axs[i][2].imshow(y_val[i])
        axs[i][2].set_title("Ground Truth")

        for ax in axs[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_val, y_val, model_name="Model"):
    print(f"\n🔍 Evaluating {model_name}...")
    psnr_list, ssim_list = [], []
    
    for i in range(len(X_val)):
        pred = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)[0]
        gt = y_val[i]

        psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim_val = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print(f"📈 {model_name} - PSNR rata-rata: {avg_psnr:.2f} dB")
    print(f"📊 {model_name} - SSIM rata-rata: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

def plot_loss(history, model_name="Model"):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - Training vs Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def demo_augmentation(clear_img, hazy_img):
    """Demonstrasi berbagai jenis augmentasi"""
    augmentor = SmartAugmentation()
    
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original
    axs[0][0].imshow(clear_img)
    axs[0][0].set_title("Original Clear")
    axs[0][1].imshow(hazy_img)
    axs[0][1].set_title("Original Hazy")
    
    # Geometric augmentation
    geo_clear = augmentor.geometric_augmentation(clear_img)
    geo_hazy = augmentor.geometric_augmentation(hazy_img)
    axs[0][2].imshow(geo_clear)
    axs[0][2].set_title("Geometric Aug Clear")
    axs[0][3].imshow(geo_hazy)
    axs[0][3].set_title("Geometric Aug Hazy")
    
    # Photometric augmentation
    photo_clear = augmentor.photometric_augmentation(clear_img)
    photo_hazy = augmentor.photometric_augmentation(hazy_img)
    axs[1][0].imshow(photo_clear)
    axs[1][0].set_title("Photometric Aug Clear")
    axs[1][1].imshow(photo_hazy)
    axs[1][1].set_title("Photometric Aug Hazy")
    
    # Haze simulation
    synthetic_hazy = augmentor.simulate_haze_variation(clear_img)
    axs[1][2].imshow(synthetic_hazy)
    axs[1][2].set_title("Synthetic Haze")
    axs[1][3].imshow(clear_img)
    axs[1][3].set_title("Original Clear")
    
    # Combined augmentation
    aug_clear, aug_hazy = augmentor.apply_all_augmentations(clear_img, hazy_img)
    axs[2][0].imshow(aug_clear)
    axs[2][0].set_title("Combined Aug Clear")
    axs[2][1].imshow(aug_hazy)
    axs[2][1].set_title("Combined Aug Hazy")
    
    # Dark channel visualization
    dark_ch = augmentor.dark_channel_prior(clear_img)
    axs[2][2].imshow(dark_ch, cmap='gray')
    axs[2][2].set_title("Dark Channel")
    axs[2][3].axis('off')
    
    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# === 9. Model Comparison Function ===
def compare_models(X_train, y_train, X_val, y_val, epochs=15):
    """Membandingkan performa CNN murni vs CNN-FNet"""
    print("🔬 Membandingkan CNN murni vs CNN-FNet...")
    
    # Data generators
    train_gen = AugmentedDataGenerator(X_train, y_train, batch_size=6, augment=True)
    val_gen = AugmentedDataGenerator(X_val, y_val, batch_size=6, augment=False)
    
    results = {}
    
    # === Train CNN-FNet Model ===
    print("\n🚀 Training CNN-FNet Model...")
    fnet_model = build_cnn_fnet_model()
    print(f"📊 CNN-FNet Total Parameters: {fnet_model.count_params():,}")
    
    early_stopping = EarlyStopping(patience=4, restore_best_weights=True, monitor='val_loss')
    
    fnet_history = fnet_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate CNN-FNet
    fnet_psnr, fnet_ssim = evaluate_model(fnet_model, X_val, y_val, "CNN-FNet")
    results['fnet'] = {'model': fnet_model, 'history': fnet_history, 'psnr': fnet_psnr, 'ssim': fnet_ssim}
    
    # === Train Pure CNN Model ===
    print("\n🚀 Training Pure CNN Model...")
    cnn_model = build_pure_cnn_model()
    print(f"📊 Pure CNN Total Parameters: {cnn_model.count_params():,}")
    
    cnn_history = cnn_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate Pure CNN
    cnn_psnr, cnn_ssim = evaluate_model(cnn_model, X_val, y_val, "Pure CNN")
    results['cnn'] = {'model': cnn_model, 'history': cnn_history, 'psnr': cnn_psnr, 'ssim': cnn_ssim}
    
    # === Comparison Summary ===
    print("\n" + "="*60)
    print("📊 PERBANDINGAN HASIL")
    print("="*60)
    print(f"CNN-FNet  -> PSNR: {fnet_psnr:.2f} dB, SSIM: {fnet_ssim:.4f}, Params: {fnet_model.count_params():,}")
    print(f"Pure CNN  -> PSNR: {cnn_psnr:.2f} dB, SSIM: {cnn_ssim:.4f}, Params: {cnn_model.count_params():,}")
    
    if fnet_psnr > cnn_psnr:
        print(f"🏆 CNN-FNet menang dengan selisih PSNR: +{fnet_psnr - cnn_psnr:.2f} dB")
    else:
        print(f"🏆 Pure CNN menang dengan selisih PSNR: +{cnn_psnr - fnet_psnr:.2f} dB")
    
    # Visualize results side by side
    show_sample_result(fnet_model, X_val, y_val, "CNN-FNet")
    show_sample_result(cnn_model, X_val, y_val, "Pure CNN")
    
    # Plot training histories
    plot_loss(fnet_history, "CNN-FNet")
    plot_loss(cnn_history, "Pure CNN")
    
    return results

# === 10. Main Program ===
if __name__ == "__main__":
    # Set random seeds untuk reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Path dataset
    CLEAR_DIR = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\SOTS\MIX\GT"
    HAZY_DIR  = r"C:\KULIAH\TEKNIK SEMESTER 6\KERJA PRAKTEK\CNNDEHAZING\SOTS\MIX\HAZY"

    # Load dataset
    print("📁 Loading dataset...")
    X, y = load_dataset(CLEAR_DIR, HAZY_DIR, limit=150)  # Kurangi dataset untuk CPU
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Demo augmentasi
    print("\n🎨 Demonstrasi augmentasi...")
    demo_augmentation(y_train[0], X_train[0])

    # Pilihan mode eksperimen
    print("\n🔬 Pilih mode eksperimen:")
    print("1. Training CNN-FNet saja (Recommended)")
    print("2. Perbandingan CNN-FNet vs Pure CNN")
    print("3. Training Pure CNN saja")
    
    try:
        choice = input("Masukkan pilihan (1/2/3): ").strip()
    except:
        choice = "1"  # Default
    
    if choice == "2":
        # === FULL COMPARISON MODE ===
        results = compare_models(X_train, y_train, X_val, y_val, epochs=12)
        
        # Save best model
        best_model = None
        best_name = ""
        if results['fnet']['psnr'] > results['cnn']['psnr']:
            best_model = results['fnet']['model']
            best_name = "CNN-FNet"
        else:
            best_model = results['cnn']['model']
            best_name = "Pure-CNN"
            
        model_path = f"best_{best_name.lower().replace('-', '_')}_dehazing_model.h5"
        best_model.save(model_path)
        print(f"\n✅ Best model ({best_name}) disimpan sebagai '{model_path}'")
        
    elif choice == "3":
        # === PURE CNN ONLY MODE ===
        print("\n🚀 Training Pure CNN Model saja...")
        train_gen = AugmentedDataGenerator(X_train, y_train, batch_size=8, augment=True)
        val_gen = AugmentedDataGenerator(X_val, y_val, batch_size=8, augment=False)
        
        model = build_pure_cnn_model()
        model.summary()
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            callbacks=[EarlyStopping(patience=4, restore_best_weights=True, monitor='val_loss')],
            verbose=1
        )
        
        # Save and evaluate
        model_path = "pure_cnn_dehazing_model.h5"
        model.save(model_path)
        print(f"\n✅ Pure CNN model disimpan sebagai '{model_path}'")
        
        show_sample_result(model, X_val, y_val, "Pure CNN")
        evaluate_model(model, X_val, y_val, "Pure CNN")
        plot_loss(history, "Pure CNN")
        
    else:
        # === CNN-FNet ONLY MODE (DEFAULT) ===
        print("\n🚀 Training CNN-FNet Model...")
        
        # Create data generators dengan batch size yang CPU-friendly
        train_gen = AugmentedDataGenerator(X_train, y_train, batch_size=4, augment=True)  # Kurangi batch size
        val_gen = AugmentedDataGenerator(X_val, y_val, batch_size=4, augment=False)

        # Build dan compile CNN-FNet model
        print("\n📦 Membuat CNN-FNet model...")
        try:
            # Coba simple FNet dulu untuk CPU
            model = build_cnn_fnet_model(use_simple_fnet=True)
            print("✅ Menggunakan Simple FNet (CPU-optimized)")
        except Exception as e:
            print(f"⚠️ Error membuat FNet model: {e}")
            print("🔄 Fallback ke Pure CNN model...")
            model = build_pure_cnn_model()
        
        print(f"\n📊 Model Summary:")
        print(f"Total Parameters: {model.count_params():,}")
        print(f"Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        # Print arsitektur singkat
        print("\n🏗️ Model Architecture:")
        for i, layer in enumerate(model.layers[:10]):  # Show first 10 layers
            try:
                if hasattr(layer, 'output_shape'):
                    shape = layer.output_shape
                elif hasattr(layer, 'output'):
                    shape = layer.output.shape if hasattr(layer.output, 'shape') else "Unknown"
                else:
                    shape = "Unknown"
                print(f"  {i+1:2d}. {layer.name:25s} -> {shape}")
            except Exception as e:
                print(f"  {i+1:2d}. {layer.name:25s} -> [Shape unavailable]")
        if len(model.layers) > 10:
            print(f"  ... ({len(model.layers)-10} more layers)")

        # Training dengan early stopping
        print(f"\n🚀 Training model dengan smart augmentation...")
        print(f"Using batch size: {train_gen.batch_size}")
        print(f"Training batches per epoch: {len(train_gen)}")
        print(f"Validation batches per epoch: {len(val_gen)}")
        
        # Callbacks untuk training yang stabil
        callbacks = [
            EarlyStopping(
                patience=6, 
                restore_best_weights=True, 
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=4,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        try:
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=25,
                callbacks=callbacks,
                verbose=1
            )

            # Save model
            model_name = "cnn_fnet_dehazing_model.h5" if "FNet" in model.name else "fallback_cnn_model.h5"
            model.save(model_name)
            print(f"\n✅ Model disimpan sebagai '{model_name}'")

            # Visualisasi dan evaluasi
            print(f"\n📊 Evaluasi {model.name}...")
            show_sample_result(model, X_val, y_val, model.name)
            evaluate_model(model, X_val, y_val, model.name)
            plot_loss(history, model.name)
            
        except Exception as training_error:
            print(f"❌ Error saat training: {training_error}")
            print("💡 Saran:")
            print("  - Kurangi batch_size lebih lanjut (ke 2 atau 1)")
            print("  - Kurangi jumlah data training")
            print("  - Gunakan model yang lebih sederhana")
            print("🔄 Program akan melanjutkan tanpa training...")
        
        # Additional analysis
        print(f"\n🔍 Analisis {model.name}...")
        if "FNet" in model.name:
            print("FNet Layer Benefits:")
            print("✓ Fourier Transform untuk global feature mixing")
            print("✓ Kompleksitas O(n log n) vs O(n²) untuk attention")
            print("✓ Non-parametrik mixing - tidak ada parameter tambahan")
            print("✓ Efisien untuk sequence/spatial mixing yang panjang")
            print("✓ Cocok untuk CPU training dengan dataset terbatas")
        else:
            print("Pure CNN Benefits:")
            print("✓ Terbukti stabil untuk dehazing tasks")
            print("✓ Local feature extraction yang kuat")
            print("✓ Memory efficient untuk CPU training")

    print("\n🎉 Eksperimen selesai!")
    print("\n📝 Catatan:")
    print("- FNet menggunakan 2D FFT untuk spatial token mixing")
    print("- Model hybrid CNN-FNet menggabungkan local convolution + global mixing")
    print("- Batch size disesuaikan untuk CPU training (6-8)")
    print("- Early stopping dan LR reduction untuk training yang stabil")
    print("- Smart augmentation meningkatkan robustness model")