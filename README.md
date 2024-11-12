```markdown
# Weather Image Classification

This repository contains a project for classifying weather images using a convolutional neural network (CNN) based on the VGG19 model trained on the ImageNet dataset. The project includes steps for data preprocessing, model training, fine-tuning, and evaluation.

## Features
- Split dataset into train, validation, and test sets.
- Use VGG19 network pre-trained on ImageNet.
- Apply data augmentation for the training set.
- Fine-tune the model by making the last two convolutional blocks trainable.
- Plot training and validation accuracy and loss.

## Requirements
- Python 3.x
- Keras
- TensorFlow
- Pillow
- Matplotlib
- splitfolders

You can install the required libraries using pip:
```bash
pip install keras tensorflow pillow matplotlib splitfolders
```

## Usage

1. **Clone the repository:**
```bash
git clone https://github.com/USERNAME/Weather-Image-Classification.git
cd Weather-Image-Classification
```

2. **Prepare your dataset:**
   Place your dataset in the `/content/dataset` directory.

3. **Split the dataset:**
   Run the script to split the dataset into train, validation, and test sets:
```python
splitfolders.ratio('/content/dataset', '/content/output_folder', seed=42, ratio=(0.8, 0.1, 0.1))
```

4. **Train the model:**
   Run the script to train and fine-tune the model:
```python
python script.py
```

5. **Plot the results:**
   The script will automatically generate plots for training and validation accuracy and loss.

## Code Overview

### `splitfolders`
Splits the dataset into train, validation, and test folders with a ratio of 0.8, 0.1, and 0.1.

### `VGG19`
Uses the VGG19 network pre-trained on the ImageNet dataset.

### `ImageDataGenerator`
Applies data augmentation to the training set and resizes images to 150x150.

### Model Architecture
- Removes the top layers of VGG19.
- Adds new dense layers with batch normalization for classification.
- Fine-tunes the last two convolutional blocks of VGG19.

### Training
Compiles and trains the model using the `rmsprop` optimizer and `categorical_crossentropy` loss.

### Evaluation
Plots training and validation accuracy and loss.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
