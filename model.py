import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def generator():
    model = keras.models.Sequential([
        layers.Dense(7 * 7 * 256, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')
    ])
    return model

def discriminator():
    model = keras.models.Sequential([
        layers.Conv2D(7, (3, 3), padding='same', input_shape=(28, 28, 1)),
        layers.Conv2D(7, (3, 3), padding='same'),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(1)
    ])
    return model

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
