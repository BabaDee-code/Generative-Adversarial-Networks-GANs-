import os
import tensorflow as tf

def download_and_prepare_dataset(dataset_name="facades"):
    """Downloads and prepares the dataset for Pix2Pix."""
    data_path = tf.keras.utils.get_file(
        origin=f"https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{dataset_name}.tar.gz",
        extract=True,
        cache_dir="./data",
    )
    dataset_dir = os.path.join(os.path.dirname(data_path), dataset_name)
    return dataset_dir

def preprocess_image(image):
    """Resizes and normalizes the image."""
    image = tf.image.resize(image, [256, 256])
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

def load_image(image_file):
    """Loads and splits an image into input and target."""
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    w = tf.shape(image)[1] // 2
    input_image = image[:, :w, :]
    target_image = image[:, w:, :]
    return preprocess_image(input_image), preprocess_image(target_image)
