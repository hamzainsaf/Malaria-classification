# Malaria-classification
his MATLAB project employs Convolutional Neural Networks (CNNs) to classify cell images into 'Uninfected' or 'Parasitized' categories. It involves preprocessing images, training a CNN model with layers designed for feature extraction and classification, and visualizing the results. 

## Setup
To set up this project, clone the repository to your local machine and ensure that your MATLAB environment is configured with the necessary toolboxes.

## Running the Script
Navigate to the script directory and run `main_script.m` from the MATLAB command window. Ensure that the paths are set correctly to the data directory.

## Image Processing
The script processes images by resizing them to 28x28 pixels and converts grayscale images to RGB by channel duplication. All images are converted to double precision for neural network processing.

## Visualization
The script provides visualization of:
- The distribution of cases (infected vs. uninfected)
- Sample images from each category

## CNN Architecture
The CNN consists of multiple layers including convolutional layers, max-pooling layers, and fully connected layers. The network configuration is optimized for this specific classification task.

## Model Training
Training is performed with the following settings:
- Optimizer: Adam
- Learning Rate: 0.01
- Epochs: 5

Progress during training is visualized in MATLAB’s training progress viewer.

## Model Usage
The trained model can classify new cell images. Instructions on loading the model and using it for predictions are provided in the script `load_model.m`.


