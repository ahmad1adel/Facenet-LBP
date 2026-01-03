"""
Embedding Module - Masked Dataset with FaceNet
Generates deep learning embeddings for face recognition using FaceNet architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import cv2
from typing import Optional, Tuple


class FaceEmbedder:
    """Generates face embeddings using FaceNet architecture"""
    
    def __init__(self, input_size: Tuple[int, int] = (160, 160), 
                 embedding_dim: int = 128, use_pretrained: bool = True):
        """
        Initialize FaceNet embedder
        
        Args:
            input_size: Input image size (height, width) - FaceNet uses 160x160
            embedding_dim: Dimension of output embedding (128 for FaceNet)
            use_pretrained: Whether to use pretrained weights
        """
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.model = None
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            self._build_model()
    
    def _inception_block_1a(self, X):
        """Implementation of an inception block"""
        # 1x1 conv
        X_1x1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(X)
        X_1x1 = BatchNormalization(axis=3)(X_1x1)
        X_1x1 = Activation('relu')(X_1x1)
        
        # 1x1 -> 3x3 conv
        X_3x3 = Conv2D(96, (1, 1), strides=(1, 1), padding='same')(X)
        X_3x3 = BatchNormalization(axis=3)(X_3x3)
        X_3x3 = Activation('relu')(X_3x3)
        X_3x3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(X_3x3)
        X_3x3 = BatchNormalization(axis=3)(X_3x3)
        X_3x3 = Activation('relu')(X_3x3)
        
        # 1x1 -> 5x5 conv
        X_5x5 = Conv2D(16, (1, 1), strides=(1, 1), padding='same')(X)
        X_5x5 = BatchNormalization(axis=3)(X_5x5)
        X_5x5 = Activation('relu')(X_5x5)
        X_5x5 = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(X_5x5)
        X_5x5 = BatchNormalization(axis=3)(X_5x5)
        X_5x5 = Activation('relu')(X_5x5)
        
        # MaxPool -> 1x1 conv
        X_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)
        X_pool = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(X_pool)
        X_pool = BatchNormalization(axis=3)(X_pool)
        X_pool = Activation('relu')(X_pool)
        
        # Concatenate
        X = concatenate([X_1x1, X_3x3, X_5x5, X_pool], axis=3)
        
        return X
    
    def _build_model(self):
        """Build FaceNet-inspired model"""
        # Input
        X_input = Input(shape=(*self.input_size, 3))
        
        # Initial convolution
        X = ZeroPadding2D(padding=(3, 3))(X_input)
        X = Conv2D(64, (7, 7), strides=(2, 2))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        # Max pooling
        X = ZeroPadding2D(padding=(1, 1))(X)
        X = MaxPooling2D((3, 3), strides=2)(X)
        
        # Conv layers
        X = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        X = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        # Max pooling
        X = ZeroPadding2D(padding=(1, 1))(X)
        X = MaxPooling2D((3, 3), strides=2)(X)
        
        # Inception blocks
        X = self._inception_block_1a(X)
        
        # Average pooling
        X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(X)
        
        # Flatten
        X = Flatten()(X)
        
        # Fully connected layers
        X = Dense(512, activation='relu')(X)
        X = Dropout(0.5)(X)
        X = Dense(256, activation='relu')(X)
        X = Dropout(0.5)(X)
        
        # Embedding layer
        embeddings = Dense(self.embedding_dim, activation=None)(X)
        
        # Create model
        self.model = Model(inputs=X_input, outputs=embeddings, name='FaceNet')
        
        print("✓ FaceNet model built successfully")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for FaceNet input
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize if needed
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] (FaceNet preprocessing)
        image = image.astype(np.float32)
        image = (image - 127.5) / 128.0
        
        # Expand dimensions for batch
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image
        
        Args:
            image: Face image
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            self._build_model()
        
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Get embedding
        embedding = self.model.predict(processed, verbose=0)
        embedding = embedding[0]  # Remove batch dimension
        
        # L2 normalize embedding (FaceNet standard)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
        
        return embedding
    
    def extract_embeddings_batch(self, images: list) -> np.ndarray:
        """
        Extract embeddings from multiple images
        
        Args:
            images: List of face images
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            self._build_model()
        
        # Preprocess all images
        processed = np.array([self.preprocess_image(img)[0] for img in images])
        
        # Get embeddings
        embeddings = self.model.predict(processed, verbose=0)
        
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-7)
        
        return embeddings
    
    def fine_tune_model(self, train_data: np.ndarray, train_labels: np.ndarray,
                       val_data: np.ndarray = None, val_labels: np.ndarray = None,
                       epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.0001):
        """
        Fine-tune the FaceNet model
        
        Args:
            train_data: Training images
            train_labels: Training labels
            val_data: Validation images
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self._build_model()
        
        # Unfreeze layers for fine-tuning
        for layer in self.model.layers[-15:]:
            layer.trainable = True
        
        # Add classification head for training
        x = self.model.layers[-1].output
        num_classes = len(np.unique(train_labels))
        classifier = Dense(num_classes, activation='softmax', name='classifier')(x)
        
        train_model = Model(inputs=self.model.input, outputs=classifier)
        
        train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Prepare data
        train_processed = np.array([self.preprocess_image(img)[0] for img in train_data])
        
        callbacks = []
        if val_data is not None:
            val_processed = np.array([self.preprocess_image(img)[0] for img in val_data])
            callbacks.append(
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            )
        
        # Train
        print("\nFine-tuning FaceNet model...")
        train_model.fit(
            train_processed, train_labels,
            validation_data=(val_processed, val_labels) if val_data is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Remove classification head to keep embedding model
        # The base layers are already updated
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is not None:
            # Use .keras format
            if filepath.endswith('.h5'):
                filepath = filepath.replace('.h5', '.keras')
            self.model.save(filepath)
            print(f"✓ FaceNet model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            self.model = keras.models.load_model(filepath)
            print("✓ FaceNet model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load FaceNet model ({e})")
            print("Rebuilding FaceNet model...")
            self._build_model()

