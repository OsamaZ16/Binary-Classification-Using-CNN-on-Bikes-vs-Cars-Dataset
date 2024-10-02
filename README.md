# Binary-Classification-Using-CNN-on-Bikes-vs-Cars-Dataset
The project is based on a Convolutional Neural Network (CNN) built with Keras/TensorFlow to classify images of bikes and cars. The model uses multiple convolutional layers, L2 regularization, and other techniques to achieve high precision and recall. Overfitting is minimized through regularization. 

## Project Overview
The project performs the following steps:

### Data Preprocessing:
Images are loaded from the dataset directory.
Invalid image files are filtered out.
Images are scaled to values between 0 and 1 to improve model performance.
### CNN Architecture:
The model consists of 3 convolutional layers with increasing filters and ReLU activation.
MaxPooling layers are used after each convolutional layer to reduce feature map dimensions.
The final layers consist of a fully connected dense layer and an output layer with sigmoid activation for binary classification.
###Training:
The model is trained using the Adam optimizer and binary cross-entropy loss.
Regularization (L2) is applied to mitigate overfitting.
###Evaluation:
The model's performance is evaluated using accuracy, precision, and recall metrics.
A confusion matrix is plotted to show classification results.
## Dataset
The dataset consists of two subdirectories, one for bikes and the other for cars. Each image is labeled according to the folder it resides in.
### Data Pipeline
The data pipeline is built using the tf.keras.preprocessing.image_dataset_from_directory function, which automatically assigns labels to images based on folder structure. The data is then split into training, validation, and testing sets.
### Metrics
The model is evaluated using the following metrics:
Precision: How many of the predicted positives are actually positive.
Recall: How many of the actual positives were correctly identified.
Accuracy: Overall correctness of the model’s predictions.
## Results
The model achieves the following results on the testing data:
Precision: 0.947
Recall: 0.9062
Accuracy: 0.9265

These metrics show that the model performs well in distinguishing between bikes and cars but can be further tuned for better recall.

## Visualizations
Loss and Accuracy Curves: The loss and accuracy over the training epochs are plotted to monitor training and validation performance.

 Confusion Matrix: A confusion matrix is generated to visualize the performance of the model on the test data.

## Usage
To run this project, ensure you have Python 3.x, TensorFlow, and the necessary libraries installed.

## Running the Model
Clone this repository.
Place your dataset under the specified directory structure.

## License
This project is open-source and available under the MIT License.
