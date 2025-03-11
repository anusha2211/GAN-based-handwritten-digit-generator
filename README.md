# Generative Adversarial Network for MNIST Digits

This project implements a Generative Adversarial Network (GAN) using TensorFlow and Keras to generate realistic handwritten digits from the MNIST dataset.

## Project Structure
```
├── train.py                  # Training script
├── generate.py               # Script to generate new images
├── model.py                  # Contains the Generator and Discriminator architecture
├── GANs.ipynb                # Colab notebook showcasing model training and results
├── requirements.txt          # List of required libraries
├── README.md                 # Project documentation
```

## Setup Instructions
1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Train the model:
```bash
python train.py
```
4. Generate new handwritten digits:
```bash
python generate.py
```

## Key Features
- **Generator**: Uses transposed convolution layers to upsample and generate realistic MNIST digits.
- **Discriminator**: Uses convolution layers to classify real vs. fake images.
- **Training**: Implemented custom training loop with gradient tape for efficient updates.
- **Performance**: Achieved stable convergence with Generator Loss ~4.3 and Discriminator Loss ~11.4.

## Results
The model successfully generates high-quality digits resembling MNIST samples, demonstrating the potential of GANs in generative modeling tasks. The `GANs.ipynb` notebook includes visual examples of generated images and training progress.

## Future Improvements
- Enhance architecture for improved stability.
- Experiment with additional datasets for broader applicability.
