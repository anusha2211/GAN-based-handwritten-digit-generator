import numpy as np
import tensorflow as tf
from model import generator
import matplotlib.pyplot as plt

# Load the generator model
generator_model = generator()
generator_model.load_weights('generator_weights.h5')

def generate_image():
    noise = np.random.randn(1, 100).astype('float32')
    generated_image = generator_model(noise)
    plt.imshow(tf.reshape(generated_image, (28, 28)), cmap='gray')
    plt.title("Generated Handwritten Digit")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    generate_image()
