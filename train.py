import numpy as np
import tensorflow as tf
from model import generator, discriminator, generator_loss, discriminator_loss
import matplotlib.pyplot as plt
import time

# Load and preprocess the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images - 127.5) / 127.5
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Instantiate models and optimizers
generator_model = generator()
discriminator_model = discriminator()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training step
@tf.function
def train_step(images):
    noise = np.random.randn(BATCH_SIZE, 100).astype('float32')

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise)

        real_output = discriminator_model(images)
        fake_output = discriminator_model(generated_images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator_model.trainable_variables))

    return gen_loss, disc_loss

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        print(f"Epoch {epoch + 1}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
        print(f'Time for epoch {epoch + 1} is {time.time() - start:.2f} sec\n')

# Train the model
train(dataset, 100)
