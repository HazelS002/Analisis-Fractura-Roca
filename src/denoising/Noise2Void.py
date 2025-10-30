import numpy as np
from tensorflow import keras

class Noise2Void:
    def __init__(self, input_shape=None, learning_rate=1e-4):
        if input_shape is not None:
            self.input_shape = input_shape
            self.model = self.build_model()
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='mse',
                metrics=['mae']
            )
        else:
            self.model = None
            self.input_shape = None
            
    def build_model(self):
        """Construye una red U-Net simplificada"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder
        x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(2)(x)
        
        x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(2)(x)
        
        # Bottleneck
        x = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        # Decoder
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        
        outputs = keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
        
        return keras.Model(inputs, outputs)
    
    def create_n2v_mask(self, batch_size, height, width, channels, blind_spot_ratio=0.2):
        """Crea máscaras aleatorias para Noise2Void"""
        masks = np.ones((batch_size, height, width, channels))
        
        for i in range(batch_size):
            # Crear máscara aleatoria donde algunos píxeles se ponen a 0
            mask = np.ones((height, width, channels))
            
            # Número de píxeles a enmascarar
            num_blind_pixels = int(height * width * blind_spot_ratio)
            
            # Seleccionar posiciones aleatorias para enmascarar
            blind_positions = np.random.choice(
                height * width, 
                size=num_blind_pixels, 
                replace=False
            )
            
            for pos in blind_positions:
                h_pos = pos // width
                w_pos = pos % width
                mask[h_pos, w_pos, :] = 0
            
            masks[i] = mask
        
        return masks
    
    def n2v_data_generator(self, noisy_images, batch_size=16, blind_spot_ratio=0.2):
        """Generador de datos corregido para Noise2Void"""
        num_samples = len(noisy_images)
        
        while True:
            # Seleccionar batch aleatorio
            indices = np.random.choice(num_samples, batch_size)
            batch_images = noisy_images[indices]
            
            batch_size_actual = len(batch_images)
            h, w, c = batch_images[0].shape
            
            # Crear máscaras de puntos ciegos
            batch_masks = self.create_n2v_mask(
                batch_size_actual, h, w, c, blind_spot_ratio
            )
            
            inputs, targets = [], []
            
            for img, mask in zip(batch_images, batch_masks):
                # Input: reemplazar píxeles enmascarados con valores aleatorios
                input_img = img.copy()
                blind_spots = (mask == 0)
                
                # Reemplazar píxeles ciegos con valores aleatorios de OTRA parte de la imagen
                if np.sum(blind_spots) > 0:
                    random_values = np.random.choice(
                        img.flatten(), 
                        size=np.sum(blind_spots)
                    )
                    input_img[blind_spots] = random_values
                
                # Target: la imagen original (el modelo debe aprender a reconstruir los píxeles enmascarados)
                target = img.copy()
                
                inputs.append(input_img)
                targets.append(target)
            
            yield np.array(inputs), np.array(targets)
    
    def train(self, noisy_images, epochs=100, batch_size=8, validation_split=0.1):
        """Entrenar el modelo Noise2Void"""
        # Verificar dimensiones
        if len(noisy_images.shape) != 4:
            raise ValueError(f"Las imágenes deben tener forma (samples, height, width, channels). Forma actual: {noisy_images.shape}")
        
        # Dividir datos
        split_idx = int(len(noisy_images) * (1 - validation_split))
        train_data = noisy_images[:split_idx]
        val_data = noisy_images[split_idx:]
        
        # Verificar que hay suficientes datos
        if len(train_data) < batch_size:
            raise ValueError(f"Batch size ({batch_size}) mayor que datos de entrenamiento ({len(train_data)})")
        
        # Generadores
        train_gen = self.n2v_data_generator(train_data, batch_size)
        val_gen = self.n2v_data_generator(val_data, batch_size)
        
        # Steps por época
        train_steps = max(1, len(train_data) // batch_size)
        val_steps = max(1, len(val_data) // batch_size)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        ]
        
        # Entrenar
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def denoise(self, noisy_images):
        """Aplicar denoising a imágenes"""
        return self.model.predict(noisy_images)
    
    def predict_single(self, noisy_image):
        """Predecir una sola imagen"""
        if len(noisy_image.shape) == 2:
            noisy_image = np.expand_dims(noisy_image, axis=-1)
        noisy_image = np.expand_dims(noisy_image, axis=0)
        denoised = self.model.predict(noisy_image)
        return denoised[0, :, :, 0]

    def save_model(self, path: str):
        self.model.save(path)
    
    def load_trained_model(self, path: str):
        self.model = keras.models.load_model(path)

if __name__ == "__main__":
    import json
    from src.utils.load_images import read_images
    from src.denoising import clean_images, get_stage_images

    print("Cargando imágenes...")
    noisy_images, _ = read_images("./data/sample-images/images/")
    results = clean_images(noisy_images)
    noisy_images = get_stage_images(results, "thresh")
    
    noisy_images = np.array([np.expand_dims(img, axis=-1) for img in noisy_images])
    print(f"Imágenes cargadas: {noisy_images.shape}\nCreando Modelo...")
    
    h, w = noisy_images[0].shape[:2]
    model = Noise2Void((h, w, 1))
    print("Modelo creado\nEntrenando...")

    # Entrenamiento
    history = model.train(noisy_images, epochs=100, batch_size=8, validation_split=0.1)
    print("Modelo entrenado\nGuardando modelo...")

    # Guardar modelo y historial de entrenamiento
    model.save_model("./models/Noise2Void/Noise2Void.h5")

    # convertir a float
    history.history = { key: [float(value) for value in values]\
                       for key, values in history.history.items() }
    with open("./models/Noise2Void/Noise2Void.json", 'w') as f:
        json.dump(history.history, f)

    print("Modelo guardado en ./models/Noise2Void/")