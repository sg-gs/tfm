#!/usr/bin/env python3
"""
Self2Self FIPS con TensorFlow/Keras
Machine Learning real - Más estable que PyTorch
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import random
from tqdm import tqdm
import os

# Configurar TensorFlow para estabilidad
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

IMAGE_PATH = "../data.fits"

CONFIG = {
    'num_epochs': 100,
    'batch_size': 16,
    'patch_size': 128,
    'learning_rate': 1e-3,
    'num_predictions': 10,
    'dropout_rate': 0.3,
}
# CONFIG = {
#     'num_epochs': 150,           # Aumentado para mejor convergencia
#     'batch_size': 32,            # Aprovechando mejor la GPU
#     'patch_size': 128,           # Mantener - funciona bien
#     'learning_rate': 2e-3,       # Ligeramente superior para convergencia más rápida
#     'num_predictions': 8,        # Reducido para eficiencia manteniendo calidad
#     'dropout_rate': 0.4,         # Aumentado para mejor generalización
# }

print(f"🧠 Self2Self FIPS con TensorFlow")
print(f"📁 Imagen: {IMAGE_PATH}")
print(f"⚡ TensorFlow: {tf.__version__}")

# =============================================================================
# MODELO SELF2SELF CON TENSORFLOW/KERAS
# =============================================================================

def create_self2self_model(input_shape=(128, 128, 4), dropout_rate=0.3):
    """
    Crear modelo Self2Self con Keras
    U-Net simple con dropout para self-supervision
    """
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck con Dropout (CLAVE para Self2Self)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(dropout_rate)(conv4)
    
    # Decoder
    up5 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(drop4)
    merge5 = layers.concatenate([conv3, up5], axis=3)
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv5)
    
    up6 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv2, up6], axis=3)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv1, up7], axis=3)
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv7)
    
    # Salida
    raw_outputs = layers.Conv2D(4, 1, activation='linear')(conv7)

    outputs = StokesConstraintLayer()(raw_outputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# =============================================================================
# RESTRICCIONES DE STOKES
# =============================================================================
class StokesConstraintLayer(layers.Layer):
    """
    Capa que aplica restricciones físicas de Stokes
    """
    def call(self, inputs):
        # Separar parámetros
        I = tf.nn.sigmoid(inputs[..., 0:1])  # I siempre positivo
        Q = tf.nn.tanh(inputs[..., 1:2])     # Q puede ser negativo
        U = tf.nn.tanh(inputs[..., 2:3])     # U puede ser negativo  
        V = tf.nn.tanh(inputs[..., 3:4])     # V puede ser negativo
        
        # Normalizar Q, U, V para cumplir restricción
        pol_magnitude = tf.sqrt(Q**2 + U**2 + V**2)
        scale_factor = tf.minimum(1.0, I / (pol_magnitude + 1e-8))
        
        Q_constrained = Q * scale_factor
        U_constrained = U * scale_factor
        V_constrained = V * scale_factor
        
        return tf.concat([I, Q_constrained, U_constrained, V_constrained], axis=-1)

# =============================================================================
# GENERADOR DE DATOS PARA SELF2SELF
# =============================================================================

class StokesDataGenerator(keras.utils.Sequence):
    """
    Generador de datos para Self2Self con parámetros de Stokes
    """
    
    def __init__(self, stokes_cube, patch_size=128, batch_size=16, patches_per_epoch=1000, **kwargs):
        super().__init__(**kwargs)  # Para evitar el warning
        
        # stokes_cube debe tener forma [H, W, 4] donde la última dimensión es [I, Q, U, V]
        self.stokes_cube = stokes_cube
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.patches_per_epoch = patches_per_epoch
        
        # Validar que tenemos los 4 parámetros de Stokes
        assert stokes_cube.shape[-1] == 4, "Se esperan 4 parámetros de Stokes [I, Q, U, V]"
        
        # Validar restricciones físicas en los datos originales
        self._validate_stokes_physics()

    def __len__(self):
        """Número de batches por época"""
        return self.patches_per_epoch // self.batch_size

    def __getitem__(self, idx):
        """Generar un batch de patches de Stokes"""
        batch_x = []
        batch_y = []
        
        for _ in range(self.batch_size):
            patch = self._get_random_patch()
            
            # Para Self2Self: input = target (misma imagen)
            batch_x.append(patch)
            batch_y.append(patch)
        
        return np.array(batch_x), np.array(batch_y)
    
    def _validate_stokes_physics(self):
        """Verificar restricciones físicas de Stokes en los datos"""
        I = self.stokes_cube[..., 0]
        Q = self.stokes_cube[..., 1]
        U = self.stokes_cube[..., 2]
        V = self.stokes_cube[..., 3]
        
        pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
        violations = np.sum(pol_intensity > I * 1.001)  # Tolerancia pequeña para errores numéricos
        
        if violations > 0:
            print(f"⚠️ Advertencia: {violations} píxeles violan restricciones de Stokes")
            # Opcionalmente, corregir automáticamente
            self._correct_stokes_violations()
    
    def _correct_stokes_violations(self):
        """Corregir violaciones de restricciones de Stokes"""
        I = self.stokes_cube[..., 0]
        Q = self.stokes_cube[..., 1]
        U = self.stokes_cube[..., 2]
        V = self.stokes_cube[..., 3]
        
        pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
        
        # Encontrar píxeles que violan la restricción
        violation_mask = pol_intensity > I
        
        if np.any(violation_mask):
            # Escalar Q, U, V para que cumplan la restricción
            scale_factor = np.ones_like(pol_intensity)
            scale_factor[violation_mask] = (I[violation_mask] * 0.99) / pol_intensity[violation_mask]
            
            self.stokes_cube[..., 1] *= scale_factor
            self.stokes_cube[..., 2] *= scale_factor
            self.stokes_cube[..., 3] *= scale_factor
            
            print(f"✅ Corregidas {np.sum(violation_mask)} violaciones de Stokes")
    
    def _get_random_patch(self):
        """
        Extraer patch aleatorio manteniendo coherencia de Stokes
        Retorna patch con forma [patch_size, patch_size, 4]
        """
        h, w, channels = self.stokes_cube.shape
        
        if h >= self.patch_size and w >= self.patch_size:
            # Imagen suficientemente grande - extraer patch aleatorio
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)
            
            # Extraer todos los canales de Stokes simultáneamente
            patch = self.stokes_cube[y:y+self.patch_size, x:x+self.patch_size, :].copy()
            
        else:
            # Imagen pequeña - usar padding
            patch = self.stokes_cube.copy()
            
            if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                pad_h = max(0, self.patch_size - patch.shape[0])
                pad_w = max(0, self.patch_size - patch.shape[1])
                
                # Aplicar padding a todos los canales manteniendo correlaciones
                patch = np.pad(patch, 
                             ((0, pad_h), (0, pad_w), (0, 0)), 
                             mode='reflect')
                
            patch = patch[:self.patch_size, :self.patch_size, :]
        
        # Aplicar augmentaciones manteniendo coherencia física
        patch = self._apply_stokes_augmentations(patch)
        
        return patch
    
    def _apply_stokes_augmentations(self, patch):
        """
        Aplicar augmentaciones respetando las propiedades físicas de Stokes
        """
        # Flip horizontal - afecta la polarización lineal
        if random.random() > 0.5:
            patch = np.fliplr(patch).copy()
            # Para flip horizontal: I y V no cambian, Q cambia signo, U no cambia
            patch[..., 1] *= -1  # Q cambia signo
        
        # Flip vertical - afecta la polarización lineal
        if random.random() > 0.5:
            patch = np.flipud(patch).copy()
            # Para flip vertical: I y V no cambian, Q no cambia, U cambia signo
            patch[..., 2] *= -1  # U cambia signo
        
        # Rotación de 90 grados - intercambia Q y U con signos apropiados
        if random.random() > 0.25:  # 25% probabilidad de rotación
            num_rotations = random.randint(1, 3)
            patch = np.rot90(patch, k=num_rotations, axes=(0, 1)).copy()
            
            # Para cada rotación de 90°:
            # I y V no cambian
            # Q y U se transforman según matriz de rotación de polarización
            for _ in range(num_rotations):
                Q_old = patch[..., 1].copy()
                U_old = patch[..., 2].copy()
                patch[..., 1] = -U_old  # Nuevo Q = -U_viejo
                patch[..., 2] = Q_old   # Nuevo U = Q_viejo
        
        # Verificar que las augmentaciones no rompan restricciones físicas
        self._verify_patch_physics(patch)
        
        return patch
    
    def _verify_patch_physics(self, patch):
        """Verificar que el patch mantiene restricciones físicas"""
        I = patch[..., 0]
        Q = patch[..., 1]
        U = patch[..., 2]
        V = patch[..., 3]
        
        pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
        violations = np.sum(pol_intensity > I * 1.001)
        
        if violations > 0:
            print(f"⚠️ Patch con {violations} violaciones después de augmentación")

def stokes_combined_loss(y_true, y_pred):
    """
    Función de pérdida combinada para parámetros de Stokes
    """
    # Pérdida MSE estándar
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Pérdida por violación de restricciones físicas
    physics_loss = stokes_physics_loss(y_true, y_pred)
    
    # Pérdida de suavidad espacial para coherencia
    smoothness_loss = stokes_spatial_smoothness_loss(y_pred)
    
    # Combinar pérdidas con pesos
    total_loss = mse_loss + 0.1 * physics_loss + 0.05 * smoothness_loss
    
    return total_loss

def stokes_physics_loss(y_true, y_pred):
    """Pérdida por violación de restricciones físicas de Stokes"""
    
    I_pred = y_pred[..., 0]
    Q_pred = y_pred[..., 1]
    U_pred = y_pred[..., 2] 
    V_pred = y_pred[..., 3]
    
    # Restricción fundamental: √(Q² + U² + V²) ≤ I
    pol_intensity = tf.sqrt(Q_pred**2 + U_pred**2 + V_pred**2 + 1e-8)
    constraint_violation = tf.maximum(0.0, pol_intensity - I_pred)
    
    # Penalizar violaciones
    physics_penalty = tf.reduce_mean(constraint_violation**2)
    
    return physics_penalty

def stokes_spatial_smoothness_loss(y_pred):
    """Pérdida de suavidad espacial para coherencia entre parámetros"""
    
    # Gradientes espaciales para cada parámetro
    gradients_loss = 0.0
    
    for i in range(4):  # Para cada parámetro de Stokes
        channel = y_pred[..., i]
        
        # Gradientes horizontales y verticales
        grad_h = channel[:, :-1, :] - channel[:, 1:, :]
        grad_v = channel[:, :, :-1] - channel[:, :, 1:]
        
        # Penalizar gradientes excesivos
        gradients_loss += tf.reduce_mean(tf.square(grad_h))
        gradients_loss += tf.reduce_mean(tf.square(grad_v))
    
    return gradients_loss / 4.0  # Normalizar por número de canales

def stokes_compliance_metric(y_true, y_pred):
    """Métrica que evalúa cumplimiento de restricciones de Stokes"""
    I_pred = y_pred[..., 0]
    pol_pred = tf.sqrt(tf.reduce_sum(y_pred[..., 1:4]**2, axis=-1))
    
    compliance_rate = tf.reduce_mean(
        tf.cast(pol_pred <= I_pred + 1e-6, tf.float32)
    )
    return compliance_rate

# =============================================================================
# CARGA Y PREPROCESAMIENTO DE IMAGEN FIPS
# =============================================================================

def load_fits_image(fits_path, observation_index=0):
    """
    Cargar cubo de datos de Stokes desde FITS
    Para datos con forma (6, 4, H, W) - seleccionar una observación
    """
    print(f"📥 Cargando cubo de Stokes: {fits_path}")
    
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    
    print(f"🔍 Datos originales: {data.shape}")
    
    # Seleccionar una observación específica
    if len(data.shape) == 4 and data.shape[0] == 6 and data.shape[1] == 4:
        print(f"   - Detectado: 6 observaciones de 4 parámetros Stokes")
        print(f"   - Seleccionando observación {observation_index}")
        
        # Tomar una observación: (4, H, W)
        single_obs = data[observation_index]
        
        # Reorganizar a (H, W, 4)
        stokes_cube = np.transpose(single_obs, (1, 2, 0))
        
    else:
        raise ValueError(f"Formato de datos inesperado: {data.shape}")
    
    print(f"✅ Cubo de Stokes final: {stokes_cube.shape}")
    
    # Verificar que los datos son físicamente válidos
    I, Q, U, V = stokes_cube[..., 0], stokes_cube[..., 1], stokes_cube[..., 2], stokes_cube[..., 3]
    pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
    violations = np.sum(pol_intensity > I * 1.001)
    
    print(f"📊 Validación física:")
    print(f"   - I rango: [{I.min():.3f}, {I.max():.3f}]")
    print(f"   - Violaciones Stokes: {violations}/{I.size}")
    
    if violations == 0:
        print(f"✅ Datos físicamente válidos")
    else:
        print(f"⚠️ {violations} violaciones detectadas")
    
    # Normalización preservando relaciones físicas
    # Normalizar basándose en el parámetro I (intensidad total)
    I_min, I_max = I.min(), I.max()
    
    if I_max - I_min > 0:
        # Normalizar todos los parámetros manteniendo relaciones
        stokes_cube_norm = stokes_cube.copy()
        stokes_cube_norm[..., 0] = (I - I_min) / (I_max - I_min)  # I normalizado [0,1]
        
        # Q, U, V normalizados manteniendo proporciones relativas a I
        I_scale = (I_max - I_min)
        stokes_cube_norm[..., 1] = Q / I_scale  # Q normalizado
        stokes_cube_norm[..., 2] = U / I_scale  # U normalizado  
        stokes_cube_norm[..., 3] = V / I_scale  # V normalizado
        
        norm_params = (I_min, I_max)
    else:
        stokes_cube_norm = stokes_cube
        norm_params = (0, 1)
    
    print(f"✅ Normalización completada")
    print(f"   - I normalizado: [{stokes_cube_norm[..., 0].min():.3f}, {stokes_cube_norm[..., 0].max():.3f}]")
    
    return stokes_cube_norm, header, norm_params


# =============================================================================
# ENTRENAMIENTO SELF2SELF
# =============================================================================

def train_self2self_tensorflow(image, config):
    """Entrenar modelo Self2Self con TensorFlow"""
    
    print(f"\n🚀 Iniciando entrenamiento Self2Self...")
    
    # Crear modelo
    model = create_self2self_model(
        input_shape=(config['patch_size'], config['patch_size'], 4),
        dropout_rate=config['dropout_rate']
    )
    
    print(f"🏗️ Modelo creado:")
    print(f"   - Parámetros: {model.count_params():,}")
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=stokes_combined_loss,
        metrics=['mae', stokes_compliance_metric]
    )
    
    # Crear generador de datos
    train_generator = StokesDataGenerator(
        stokes_cube=image,
        patch_size=config['patch_size'],
        batch_size=config['batch_size'],
        patches_per_epoch=2000
    )
    
    print(f"📊 Configuración entrenamiento:")
    print(f"   - Épocas: {config['num_epochs']}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Patches por época: 2000")
    print(f"   - Batches por época: {len(train_generator)}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss', patience=20, restore_best_weights=True, verbose=1
        )
    ]
    
    # Entrenar
    print(f"\n⏱️ Entrenamiento iniciado...")
    
    history = model.fit(
        train_generator,
        epochs=config['num_epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"✅ Entrenamiento completado!")
    
    # Guardar modelo
    model.save('self2self_fips_tensorflow.h5')
    print(f"💾 Modelo guardado: self2self_fips_tensorflow.h5")
    
    return model, history


# =============================================================================
# INFERENCIA SELF2SELF
# =============================================================================

def denoise_with_self2self(model, image, config):
    """Aplicar denoising Self2Self con múltiples predicciones para datos de Stokes"""
    
    print(f"\n🧠 Aplicando Self2Self denoising...")
    print(f"   - Imagen: {image.shape}")
    print(f"   - Predicciones: {config['num_predictions']}")
    
    h, w, channels = image.shape  # CAMBIO: ahora maneja 3 dimensiones
    patch_size = config['patch_size']
    
    if h <= patch_size and w <= patch_size:
        # Imagen pequeña - procesar completa
        print("   - Modo: Cubo completo")
        
        # Preparar imagen
        img_input = image.copy()
        
        # Hacer padding si es necesario
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img_input = np.pad(img_input, 
                              ((0, pad_h), (0, pad_w), (0, 0)), 
                              mode='reflect')
        
        img_input = img_input[:patch_size, :patch_size, :]
        img_input = np.expand_dims(img_input, 0)  # Añadir dimensión de batch
        
        # Múltiples predicciones con dropout
        predictions = []
        for i in range(config['num_predictions']):
            pred = model(img_input, training=True)
            predictions.append(pred.numpy().squeeze())
        
        # Promediar predicciones
        result = np.mean(predictions, axis=0)
        result = result[:h, :w, :]  # Recortar al tamaño original
        
    else:
        # Imagen grande - procesar por patches
        print("   - Modo: Por patches")
        result = denoise_by_patches(model, image, config)
    
    return result


def denoise_by_patches(model, image, config):
    """Denoising por patches para datos de Stokes grandes"""
    
    h, w, channels = image.shape  # CAMBIO: 3 dimensiones
    patch_size = config['patch_size']
    overlap = 32
    
    # Posiciones de patches
    y_positions = list(range(0, h - patch_size + 1, patch_size - overlap))
    x_positions = list(range(0, w - patch_size + 1, patch_size - overlap))
    
    if y_positions[-1] + patch_size < h:
        y_positions.append(h - patch_size)
    if x_positions[-1] + patch_size < w:
        x_positions.append(w - patch_size)
    
    result = np.zeros_like(image)
    weights = np.zeros((h, w))  # Solo 2D para pesos
    
    total_patches = len(y_positions) * len(x_positions)
    patch_count = 0
    
    print(f"      Total patches: {total_patches}")
    
    for y in y_positions:
        for x in x_positions:
            # Extraer patch de todos los canales
            patch = image[y:y+patch_size, x:x+patch_size, :].copy()
            patch_input = np.expand_dims(patch, 0)
            
            # Múltiples predicciones
            patch_predictions = []
            for i in range(config['num_predictions']):
                pred = model(patch_input, training=True)
                patch_predictions.append(pred.numpy().squeeze())
            
            denoised_patch = np.mean(patch_predictions, axis=0)
            
            # Peso para blending
            weight_patch = create_weight_patch(patch_size, overlap)
            
            # Acumular resultado con pesos por canal
            for c in range(channels):
                result[y:y+patch_size, x:x+patch_size, c] += denoised_patch[:, :, c] * weight_patch
            weights[y:y+patch_size, x:x+patch_size] += weight_patch
            
            patch_count += 1
            if patch_count % 20 == 0:
                print(f"      Procesados {patch_count}/{total_patches}")
    
    # Normalizar por pesos
    final_result = np.zeros_like(image)
    for c in range(channels):
        final_result[:, :, c] = np.divide(
            result[:, :, c], 
            weights, 
            out=np.zeros_like(weights), 
            where=weights != 0
        )
    
    return final_result


def create_weight_patch(patch_size, overlap):
    """Crear patch de pesos para blending"""
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    
    if overlap > 0:
        fade = np.linspace(0, 1, overlap)
        weight[:overlap, :] *= fade[:, np.newaxis]
        weight[-overlap:, :] *= fade[::-1, np.newaxis]
        weight[:, :overlap] *= fade[np.newaxis, :]
        weight[:, -overlap:] *= fade[::-1][np.newaxis, :]
    
    return weight


# =============================================================================
# UTILIDADES PARA STOKES
# =============================================================================

def desnormalize_stokes(stokes_normalized, I_min, I_max):
    """Desnormalizar cubo de Stokes manteniendo relaciones físicas"""
    
    stokes_original = stokes_normalized.copy()
    I_scale = I_max - I_min
    
    # Desnormalizar I
    stokes_original[..., 0] = stokes_normalized[..., 0] * I_scale + I_min
    
    # Desnormalizar Q, U, V 
    stokes_original[..., 1] = stokes_normalized[..., 1] * I_scale
    stokes_original[..., 2] = stokes_normalized[..., 2] * I_scale
    stokes_original[..., 3] = stokes_normalized[..., 3] * I_scale
    
    return stokes_original

def validate_stokes_physics_post_denoising(stokes_cube):
    """Validar que el cubo denoised mantiene física de Stokes"""
    
    I = stokes_cube[..., 0]
    Q = stokes_cube[..., 1]
    U = stokes_cube[..., 2]
    V = stokes_cube[..., 3]
    
    # Verificar restricciones
    pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
    violations = np.sum(pol_intensity > I * 1.001)
    total_pixels = I.size
    
    compliance_rate = 1.0 - (violations / total_pixels)
    
    print(f"\n🔬 Validación física post-denoising:")
    print(f"   - Violaciones: {violations}/{total_pixels}")
    print(f"   - Cumplimiento: {compliance_rate:.4f} ({compliance_rate*100:.2f}%)")
    print(f"   - I rango: [{I.min():.4f}, {I.max():.4f}]")
    print(f"   - |Q| máximo: {np.abs(Q).max():.4f}")
    print(f"   - |U| máximo: {np.abs(U).max():.4f}")  
    print(f"   - |V| máximo: {np.abs(V).max():.4f}")
    
    return compliance_rate > 0.95


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def create_comparison(original, denoised, save_path='stokes_comparison.png'):
    """Crear comparación específica para parámetros de Stokes"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    stokes_labels = ['I (Intensidad)', 'Q (Pol. Linear)', 'U (Pol. Linear)', 'V (Pol. Circular)']
   
    for i in range(4):
        # Original
        im1 = axes[0, i].imshow(original[..., i], cmap='hot' if i==0 else 'coolwarm', 
                                origin='lower')
        axes[0, i].set_title(f'Original {stokes_labels[i]}', fontweight='bold')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
        
        # Denoised
        im2 = axes[1, i].imshow(denoised[..., i], cmap='hot' if i==0 else 'coolwarm', 
                                origin='lower')
        axes[1, i].set_title(f'Self2Self {stokes_labels[i]}', fontweight='bold')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparación Stokes guardada: {save_path}")


def print_stokes_statistics(original, denoised):
    """Imprimir estadísticas específicas para Stokes"""
    
    print("\n📈 ESTADÍSTICAS FINALES STOKES:")
    print("=" * 40)
    
    stokes_names = ['I', 'Q', 'U', 'V']
    
    for i, name in enumerate(stokes_names):
        orig_std = np.std(original[..., i])
        denoised_std = np.std(denoised[..., i])
        noise_reduction = (orig_std - denoised_std) / orig_std if orig_std > 0 else 0
        
        mse = np.mean((original[..., i] - denoised[..., i])**2)
        
        print(f"✅ {name}: reducción ruido = {noise_reduction:.1%}, MSE = {mse:.6f}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Pipeline completo Self2Self con TensorFlow"""
    
    print("🧠 SELF2SELF STOKES CON TENSORFLOW")
    print("=" * 50)
    
    # Verificar imagen
    if not Path(IMAGE_PATH).exists():
        print(f"❌ Error: No se encuentra {IMAGE_PATH}")
        return
    
    try:
        # 1. Cargar imagen
        print("\n📊 PASO 1: Cargando imagen...")
        image, header, norm_params = load_fits_image(IMAGE_PATH)
        
        # 2. Entrenar modelo
        print("\n🎯 PASO 2: Entrenando modelo Self2Self...")
        model, history = train_self2self_tensorflow(image, CONFIG)
        
        # 3. Aplicar denoising
        print("\n🧠 PASO 3: Aplicando denoising...")
        denoised = denoise_with_self2self(model, image, CONFIG)
        
        # 4. Validar física post-denoising
        physics_ok = validate_stokes_physics_post_denoising(denoised)
        if not physics_ok:
            print("⚠️ Advertencia: Problemas de cumplimiento físico detectados")
        
        # 5. Desnormalizar
        I_min, I_max = norm_params
        if I_min != 0 or I_max != 1:
            denoised_original = desnormalize_stokes(denoised, I_min, I_max)
            original_scale = desnormalize_stokes(image, I_min, I_max)
        else:
            denoised_original = denoised
            original_scale = image
        
        # 6. Guardar resultado
        print("\n💾 PASO 4: Guardando resultado...")
        
        # Reorganizar para FITS: (H, W, 4) -> (4, H, W)
        denoised_fits_format = np.transpose(denoised_original, (2, 0, 1))
        
        hdu = fits.PrimaryHDU(denoised_fits_format.astype(np.float32))
        if header:
            hdu.header.update(header)
        hdu.writeto('fips_denoised_tensorflow.fits', overwrite=True)
        print("✅ Resultado guardado: fips_denoised_tensorflow.fits")
        
        # 7. Crear visualización
        print("\n📊 PASO 5: Creando visualización...")
        create_comparison(original_scale, denoised_original)
        
        # 8. Estadísticas
        print_stokes_statistics(original_scale, denoised_original)
        
        # Estadísticas adicionales de compatibilidad
        print("\n📈 RESULTADOS ADICIONALES:")
        print("=" * 30)
        
        # Calcular métricas por canal
        for i, name in enumerate(['I', 'Q', 'U', 'V']):
            mse = np.mean((original_scale[..., i] - denoised_original[..., i])**2)
            if mse > 0:
                psnr = 20 * np.log10(original_scale[..., i].max() / np.sqrt(mse))
                print(f"✅ {name} PSNR: {psnr:.1f} dB")
        
        print(f"✅ Loss final: {history.history['loss'][-1]:.6f}")
        if 'stokes_compliance_metric' in history.history:
            print(f"✅ Cumplimiento Stokes final: {history.history['stokes_compliance_metric'][-1]:.4f}")
        
        print(f"\n🎉 PROCESO STOKES COMPLETADO!")
        print("📁 Archivos generados:")
        print("   - self2self_fips_tensorflow.h5 (modelo entrenado)")
        print("   - fips_denoised_tensorflow.fits (cubo Stokes denoised)")
        print("   - stokes_comparison.png (comparación)")
        
        return model, original_scale, denoised_original
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# EJECUTAR
# =============================================================================

if __name__ == "__main__":
   
    # Verificar TensorFlow
    try:
        print(f"✅ TensorFlow {tf.__version__} disponible")
        
        # Test básico
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.reduce_mean(x)
        print(f"✅ Operaciones TensorFlow funcionan")
        
    except Exception as e:
        print(f"❌ Error TensorFlow: {e}")
        print("📦 Instalar: pip install tensorflow")
        exit(1)
    
    # Ejecutar
    result = main()
