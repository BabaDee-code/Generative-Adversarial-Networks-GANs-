# Pix2Pix GAN Implementation

This repository contains an implementation of the Pix2Pix Generative Adversarial Network (GAN) using TensorFlow. The Pix2Pix model is used for image-to-image translation tasks, such as converting sketches to photos, day to night images, etc.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [References](#references)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/pix2pix-gan.git
    cd pix2pix-gan
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv .GANs_env
    source .GANs_env/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To train the Pix2Pix model, you can run the [main.py](http://_vscodecontentref_/0) or [pix2pix_main.py](http://_vscodecontentref_/1) script. Both scripts perform the same function, but [pix2pix_main.py](http://_vscodecontentref_/2) is more modular.

```sh
python main.py