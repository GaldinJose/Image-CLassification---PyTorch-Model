# Image-CLassification---PyTorch-Model


## Playing Cards Image Classification

## Project Overview

This project implements an image classification model to identify playing cards from images. Utilizing PyTorch and the EfficientNet-B0 architecture from the `timm` library, the code demonstrates the process of data preparation, model training, and evaluation for classifying images into one of 53 categories, including various card ranks and suits as well as a joker.

## Key Features

- **Data Handling**: 
  - Unzips and extracts a dataset of playing card images.
  - Custom `PlayingCards` dataset class for loading and transforming images.

- **Data Preprocessing**:
  - Resizes images to 128x128 pixels.
  - Converts images to tensor format for model input.

- **Model Architecture**:
  - Defines a `CardClassifier` model using EfficientNet-B0 for feature extraction and a custom linear classifier for classification.

- **Training**:
  - Trains the model with CrossEntropyLoss and Adam optimizer.
  - Includes a simple training loop with loss tracking for both training and validation phases.

- **Evaluation**:
  - Visualizes training and validation losses over epochs.
  - Evaluates the model on a test set to compute accuracy.

- **Prediction and Visualization**:
  - Provides functions to preprocess images, make predictions, and visualize results with probability distributions for each class.

## Dataset

- **Playing Cards Dataset**: 
  - The dataset consists of images of playing cards categorized into 53 classes, including various ranks (Ace, 2, 3, etc.) and suits (Clubs, Diamonds, Hearts, Spades), plus a Joker.

## Code Walkthrough

1. **Data Extraction**:
   - Extracts images from a ZIP file into a directory structure.
   - Ensure your dataset zip file is named Cards_dataset.zip and placed in the root directory.

2. **Dataset Class**:
   - Implements `PlayingCards` class using PyTorch's `Dataset` for managing image data.

3. **Transformations**:
   - Applies resizing and tensor conversion to prepare images for the model.

4. **Model Definition**:
   - Uses EfficientNet-B0 for feature extraction with a custom classifier to predict card categories.

5. **Training Loop**:
   - Trains the model over a specified number of epochs, tracking loss for both training and validation datasets.

6. **Evaluation**:
   - Computes accuracy on the test set and plots training/validation losses.

7. **Prediction and Visualization**:
   - Implements functions to preprocess images, predict card classes, and visualize predictions with probabilities.

## Libraries Used

- **PyTorch**: For model building, training, and evaluation.
- **timm**: Provides pre-trained EfficientNet models.
- **PIL**: For image processing.
- **Matplotlib**: For plotting training/validation loss and visualizing predictions.
- **NumPy**: For numerical operations and data handling.

## Future Enhancements

- **Model Improvement**: Experiment with other architectures or fine-tune the current model further.
- **Data Augmentation**: Apply techniques such as rotation or flipping to improve model robustness.
- **Hyperparameter Tuning**: Explore different learning rates and optimization strategies.

## Contributing

Contributions to improve the model or add new features are welcome. Please submit pull requests or open issues to discuss any proposed changes.


## Acknowledgements

- **Video Tutorial**: Special thanks to the [YouTube video](https://youtu.be/tHL5STNJKag?si=XAPdhEK_msMjvOEn) for providing valuable insights and instructions that guided the development of this project.

The dataset used in this project is available [here](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification).
