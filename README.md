Capstone Project: Machine Learning Model Development
Overview
This repository contains code for developing a deep learning model for image classification using the TensorFlow and Keras libraries. The model is designed to classify images into 22 different classes, including various types of batik and non-batik categories. The project includes data preprocessing, model training, evaluation, and deployment.

Folder Structure
Dataset: Contains the training, validation, and test datasets for the machine learning model.
Notebooks: Jupyter notebooks for different stages of the project, including data exploration, model development, and evaluation.
Models: Saved models in h5 format.
Results: Output files, plots, and evaluation results.
Scripts: Python scripts for specific tasks or utilities.
Requirements
Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
scikit-learn
Install dependencies using:

bash
Copy code
pip install -r requirements.txt
Data
The dataset is organized into training, validation, and test sets. Each set contains images of various batik types and non-batik images.

Data Augmentation
Data augmentation techniques, such as rotation, shifting, and flipping, are applied to the training set to enhance model generalization.

Model Architecture
The machine learning model is built on top of the MobileNet architecture, a pre-trained model on the ImageNet dataset. Custom layers are added for fine-tuning the model on the specific task of batik classification.

Custom Metric
The model is compiled with a custom confidence metric, which measures the maximum confidence score for each prediction.

Training
The model is trained on the augmented training set with early stopping to prevent overfitting. Training progress and performance metrics are visualized using Matplotlib.

Evaluation
The model is evaluated on the test set, and performance metrics such as accuracy, precision, recall, and F1 score are computed. Confusion matrix visualizations provide insights into the model's classification performance.

Model Deployment
The trained model is saved in the 'Models' folder and can be used for making predictions on new images.
