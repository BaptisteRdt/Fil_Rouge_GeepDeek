import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class uNet(Model):
    def __init__(self, img_shape: tuple = (360, 360, 3)):
        super(uNet, self).__init__()
        self.inputs = tf.keras.layers.Input(shape=img_shape)

        # Contracting path (way to bottleneck)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.inputs)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2))(c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
        p2 = tf.keras.layers.MaxPooling2D((2))(c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.1)(c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2))(c3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.1)(c4)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2))(c4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.1)(c5)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

        # Bottleneck reached, upgrading path

        u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.1)(c6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.1)(c7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(c9)

        self.outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def call(self, x):
        return self.model(x)

    def summary(self):
        return self.model.summary()