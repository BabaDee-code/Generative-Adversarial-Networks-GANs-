import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, ReLU, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from data_utils import download_and_prepare_dataset, load_image
from model_utils import create_generator, create_discriminator, downsample, upsample
from train_utils import train_step, fit, generator_loss, discriminator_loss

if __name__ == "__main__":
    # Set up dataset
    dataset_dir = download_and_prepare_dataset()
    print(f"Dataset downloaded and prepared at: {dataset_dir}")

    train_dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, "train/*.jpg"))
    train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(400).batch(1)

    # Define and build models
    generator = create_generator()
    discriminator = create_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    print("Starting training...")
    fit(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, epochs=50)

    generator.save("generator_model.h5")
    print("Generator model saved.")