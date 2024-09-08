# Medical Image Classification with Convolutional Neural Networks


# Overview and Purpose

In this document, we train a convolutional neural network (CNN) model from scratch to classify CT chest scan images as either cancerous or normal, and then investigate the added benefit of using pre-trained CNN models on the same dataset by employing transfer learning. Furthermore, we investigate model ensembling in general and its potential benefits and drawbacks. 


## Convolutional Neural Network: General Overview


1. Input Layer: The process starts with an image input, which is typically represented as a matrix of pixel values. For a color image, this matrix would have three channels (Red, Green, Blue).

2. Convolutional Layers: These layers apply convolutional filters (or kernels) to the input image. Each filter scans through the image and performs a convolution operation, which involves element-wise multiplication and summing up the results. This process extracts features like edges, textures, and patterns. The result is a feature map that highlights the presence of specific features in different parts of the image.

3. Activation Function: After convolution, an activation function (often ReLU, or Rectified Linear Unit) is applied to introduce non-linearity into the model. This helps the network learn more complex patterns.

4. Pooling Layers: Pooling (often max pooling) is used to reduce the spatial dimensions of the feature maps. It involves taking the maximum or average value from a subset of the feature map, which reduces the number of parameters and computations, and helps the network become more robust to variations in the image.

5. Flattening: The output from the convolutional and pooling layers is a multi-dimensional tensor. This tensor is flattened into a one-dimensional vector to feed into fully connected layers.

6. Fully Connected Layers: These layers are similar to traditional neural networks where each neuron is connected to every neuron in the previous layer. They combine the features learned by the convolutional and pooling layers to make a final prediction.

7. Output Layer: The final layer is a softmax (or sigmoid) function that outputs probabilities for each class, indicating the network's prediction for the image category.

In summary, CNNs use convolutional and pooling layers to automatically and hierarchically learn features from images, followed by fully connected layers to classify those features into predefined categories. This process enables CNNs to effectively handle and classify complex visual data.

The data for this project was obtained here: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images



From the Kaggle notebook available here: https://www.kaggle.com/code/prthmgoyl/ensemblemodel-ctscan  we have some code to work with for pre-trained models. We evaluate each model on test loss, test accuracy, validation loss and validation accuracy. Below are two tables, each populated using a different type of loss function:



## Table of results using CategoricalCrossEntropy loss function and class_mode='categorical' in data generator functions

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| Base cnn  | 9.0188  | 0.1876  | 8.0590 | 0.2917 |
| EfficientNetB3  | 0.0544 | 0.9804 | 0.8051 | 0.8444 |
| ResNet50 | 0.0118 | 0.9951 | 2.0054 | 0.7587 |
| InceptionV3 | 0.0439 | 0.9886 | 2.8829 | 0.5016 |
| Ensemble | 0.3994 | 0.8750 | 0.2605 | 0.9750 |





## Table of results using SparseCategoricalCrossEntropy loss function and class_mode='sparse' in data generator functions

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| Base cnn  | 0.0112  | 0.9984  | 2.0032 | 0.6111 |
| EfficientNetB3  | 0.0536 | 0.9837 | 0.6506 | 0.8317 |
| ResNet50 | 0.0682 | 0.9886 | 2.0240 | 0.7365 |
| InceptionV3 | 0.0615 | 0.9902 | 2.9775 | 0.5111 |
| Ensemble | 0.9132 | 0.8889 | 0.9594 | 0.8999 |



## Sparse categorical vs categorical loss functions / sparse vs non-sparse class mode generators

Use the Sparse categorical crossentropy loss function when there are two or more label classes. In our data generating code, we can specify class_mode='sparse' to get the correct format in the generated data. We expect labels to be provided as integers. If one wants to provide labels using one-hot representation, please use CategoricalCrossentropy loss (with class_mode='categorical' in the data generating code instead). There should be # classes floating point values per feature for y_pred and a single floating point value per feature for y_true. In our instance above, we do not discern a noticable difference in the accuracies for the two approaches, though we do obtain the highest validation accuracy when using categorical cross entropy / categorical data generation.

Unfortunately, there is no good way to tell whether we are dealing with sparse or categorical data generation just by looking at the data vectors themselves. The two are basically indistinguishable:

### Sparse:

![image](https://github.com/schr0841/cnn_project/blob/main/images/labels_sparse.png)



### Categorical:

![image](https://github.com/schr0841/cnn_project/blob/main/images/labels_categorical.png)

Therefore, we must carefully specify whether we are using sparse or categorical in the data generating functions to ensure that everything matches up with the specified loss function.

## Pre-training a model: General Overview

Pre-training a model in the context of neural networks involves training a model on a large dataset before fine-tuning it on a specific task (the end result is known as **transfer learning**). Here’s a breakdown of what pre-training means and why it’s beneficial:

1. **Initial Training on a Large Dataset**: Pre-training typically involves training a neural network on a broad, general-purpose dataset. For example, in the case of image classification, a model might be pre-trained on a large and diverse dataset like ImageNet, which contains millions of labeled images across thousands of categories.

2. **Learning General Features**: During pre-training, the model learns to identify and extract general features from the data, such as edges, textures, and shapes in images, or basic linguistic patterns in text. These features are broadly useful across different tasks and domains.

3. **Transfer Learning**: After pre-training, the model is adapted to a specific task or dataset in a process known as transfer learning. Here, the model's weights, which have been optimized during pre-training, are used as the starting point for training on a new, often smaller, dataset. The model is fine-tuned to learn the specifics of the new task while leveraging the general features it has already learned. In our specific case, the smaller dataset consists of CT-Scan images with different types of chest cancer.

4. **Fine-Tuning**: Fine-tuning involves adjusting the pre-trained model's weights to better fit the new task. This can involve retraining some or all of the network's layers, depending on how similar the new task is to the original one.

Benefits of Pre-Training:

* **Improved Performance**: Pre-training allows the model to leverage knowledge learned from a large and diverse dataset, which can lead to better performance on the new task, especially when the new dataset is small or lacks diversity.

* **Reduced Training Time**: Training a model from scratch can be computationally expensive and time-consuming. Pre-training on a large dataset and then fine-tuning can significantly reduce the time required to achieve good performance.

* **Better Generalization**: Pre-trained models often generalize better to new tasks because they start with a solid understanding of basic features and patterns, which can help improve accuracy on the new task.

Pre-training is a powerful technique, especially in scenarios where data is scarce or where training a model from scratch would be impractical due to resource constraints.

In our specific case, we use models that are pre-trained on the EfficientNetB3, ResNet50 and InceptionV3 datasets. The InceptionV3 model is trained to classify 1000 different images in a wide range of categories, so that may be why the accuracy suffers, but the ResNet50 model also does this and has good accuracy for us. We need some further investigation as to why there are differences in these models. After pre-training, we then employ transfer learning by further training the models on our cancer image dataset.




## Base CNN architecture:

![image](https://github.com/schr0841/cnn_project/blob/main/images/base1.png)

![image](https://github.com/schr0841/cnn_project/blob/main/images/base3.png)

![image](https://github.com/schr0841/cnn_project/blob/main/images/base2.png)


## EfficientNet architecture:

![image](https://github.com/schr0841/cnn_project/blob/main/images/EfficientNet.png)



## ResNet50 architecture:

![image](https://github.com/schr0841/cnn_project/blob/main/images/ResNet.png)


## InceptionV3 architecture:

![image](https://github.com/schr0841/cnn_project/blob/main/images/Inception.png)




## Ensemble models: General Overview


Ensemble learning uses multiple machine learning models to try to make better predictions on a dataset. An ensemble model works by training different models on a dataset and having each model make predictions individually. The predictions of these models are then combined in the ensemble model to make a final prediction.

Every model has its strengths and weaknesses. Ensemble models can be beneficial by combining individual models to help improve upon the predictive performance of each individual model. 

Ensemble models are a way to improve the performance of machine learning algorithms by combining the predictions of multiple models. The main idea is that different models might make different kinds of errors, so combining them can reduce the overall error rate. Here’s a breakdown of how ensemble models work and some common techniques:

### How Ensembles Work

1. **Training Multiple Models**: First, you train multiple base models (often called “learners”) on the same problem. These models could be of the same type (e.g., multiple decision trees) or different types (e.g., decision trees, support vector machines, and neural networks).

2. **Combining Predictions**: Once trained, the ensemble combines the predictions of the base models to make a final prediction. The combination can be done in various ways, such as averaging the predictions or using a majority vote.

3. **Reducing Overfitting**: By combining models, ensembles can reduce the risk of overfitting to the training data, as errors made by individual models might be corrected by others in the ensemble.

### Common Techniques

1. **Bagging (Bootstrap Aggregating)**:
   - **Process**: Multiple versions of a model are trained on different subsets of the training data, typically created by sampling with replacement (bootstrap samples). 
   - **Example**: Random Forests, which are ensembles of decision trees.
   - **How it Helps**: Reduces variance by averaging predictions or using majority voting, leading to more stable predictions.

2. **Boosting**:
   - **Process**: Models are trained sequentially. Each new model tries to correct the errors made by the previous models. Weights are adjusted to pay more attention to the misclassified examples.
   - **Example**: AdaBoost, Gradient Boosting Machines (GBM), XGBoost.
   - **How it Helps**: Reduces bias and can convert weak learners into strong learners by focusing on errors of previous models.

3. **Stacking (Stacked Generalization)**:
   - **Process**: Different models are trained on the same data, and their predictions are used as inputs to a final model (meta-model) that makes the ultimate prediction.
   - **Example**: Combining logistic regression, decision trees, and neural networks as base models, with a meta-model like a logistic regression model that combines their predictions.
   - **How it Helps**: Leverages the strengths of different models and combines their predictions to potentially achieve better performance than any individual model.

4. **Voting**:
   - **Process**: Each model in the ensemble makes a prediction, and the final prediction is made by taking a majority vote (for classification) or averaging (for regression).
   - **Example**: Simple majority vote among different classifiers.
   - **How it Helps**: Simple to implement and can improve robustness by reducing the likelihood of errors from any single model.

### Advantages of Ensemble Models

- **Improved Accuracy**: By combining models, ensembles often achieve better accuracy than individual models.
- **Reduced Overfitting**: Ensembles can mitigate overfitting by smoothing out errors and variances.
- **Robustness**: They are more robust to changes in the input data and can handle noisy data better.

### Disadvantages of Ensemble Models

- **Increased Complexity**: Ensembles can be more complex to implement and interpret compared to individual models.
- **Computational Cost**: Training multiple models can be computationally expensive and require more resources.

Ensemble methods are a powerful tool in machine learning, often used in practice to achieve higher performance and more reliable predictions. In our specific scenario, the ensemble model uses averaging to select the mode, or the most common predicted class, from the three pre-trained models. By doing this, it allows for further generalization accuracy improvements on the unseen validation data. 



## Ensemble (basic mean/mode) architecture:

![image](https://github.com/schr0841/cnn_project/blob/main/images/ensemble.png)



## Ensemble validation set - figure out why only 1 epoch / why validation accuracy not showing up

The **test_steps** parameter in the call to model.evaluate is currently evaluating to 1, so that is why there is only 1 epoch. It seems to be related to the data generator function that we are using - when this parameter was manually increased to 10, a warning was obtained that said the input ran out of data.

The validation accuracy was originally not computed, but we added an appropriate line of code and printed these results to the console. We were then able to add the appropriate result to the table. As we expect, the validation accuracy is much higher than the individual accuracy of the three pre-trained models. 



## Exploration of 2-step boosting ensemble procedure

We attempt to train 2 models sequentially, where the second model learns from the errors of the first model. Since our best-performing models are EfficientNetB3 and ResNet50, we will use those models to form our boosted ensemble. We copy the code for these models into a new cell. However we are unable to get it to work correctly, and are leaving this as a 'future improvement' that could be worked on. 

We asked Claude AI if it could come up with an implementation of simple boosted ensemble models and it was able to do that, but the code it provided assumed we have a clear split among the features $X$ and target $y$, whereas here we do not. We have included it as an extra supplementary notebook titled "gemini_boosted_ensemble.ipynb" to look at for further development.


## Comparison of Chaining and Ensembling Models

We trained the custom_cnn model and the ResNet50 model on the CT chest scan images in the training_set directory. This directory contained 613 files belonging to four classes. Similarly, a testing_set of 315 files belonging to four classes and a validation_set of 72 files belonging to 4 classes, were created for use in validating the model during training, and for evaluating the model after training. One class pertained to images without cancer. Three classes pertained to one each of three forms of cancer. 

The data sets were generated using the tf.keras.preprocessing.image_dataset_from_directory method. This is to say they were not generated using the ImageDataGenerator, as datasets elsewhere in this study were generated. Also of note, the image_size was set to (224, 224) in both the custom_cnn_model and the ResNet50 model, because the ResNet50 model expects images of that size. For purposes of consistency, the image_size was set to (224, 224) for the custom_cnn_model as well. 
label_mode for the three datasets were set to "int" (integer) because images belong to one of four classes. 

![Screenshot 2024-09-08 163909](https://github.com/user-attachments/assets/f2faf09f-0a55-453d-ad60-ba0415310570)

![Screenshot 2024-09-08 165703](https://github.com/user-attachments/assets/b50f6779-3154-4ae8-b0e3-5f56c59ab384)
![Screenshot 2024-09-08 165747](https://github.com/user-attachments/assets/619dfd0f-6093-42d9-976f-7b7a5a5d4984)

Both the models were compiled with the Adam optimizer, and with the loss function set to sparse_categorical_crossentropy. Both were trained with x as the training_set dataset, with validation_data specified as the validation_set dataset, on 100 epochs, and with an EarlyStopping callback set to monitor 'val_accuracy' with a patience value of 20.

<img width="910" alt="Screenshot 2024-09-08 170917" src="https://github.com/user-attachments/assets/71cbc01a-20d8-45b6-afd3-498fe9fff535">
<img width="905" alt="Screenshot 2024-09-08 171613" src="https://github.com/user-attachments/assets/bc07814e-888f-4779-a562-1d874fa9ddef">
<img width="772" alt="Screenshot 2024-09-08 171817" src="https://github.com/user-attachments/assets/e91235d6-b62d-407c-bd00-172ee7cf0181">

To prevent the ResNet50 model from generating a classification for the images in the input dataset, we set include_top=False. The second model in the ensemble, the custom_cnn_model, will make the classification. To prevent the ResNet50 model from being re-trained from its ImageNet data source, we froze its layers with by specifying layer.trainable = False. 

Finally, we added some custom layers to the ResNet50 base_model before compiling and training it. 
ResNet50, when its top layer is excluded, outputs a feature map with shape (7, 7, 2048) It is not designed to classify only four classes of images. Adding custom layers to ResNet50 allows us to adapt the pretrained model to fit our specific needs (e.g., completing a four-class classification task, ensembling with the custom_cnn_model, and chaining with the custom_cnn_model). Furthermore, the added BatchNormalization and Dropout layers assist with regularizing the model, or improving its generalization on unseen data. At the same time, the custom Dense(256) layer reduces the dimensionality of the original ResNet50 model's output, making it more managable for the final output layer which outputs probabilities for each class.   



![Screenshot 2024-09-08 172623](https://github.com/user-attachments/assets/12018eb2-0e74-405c-935b-cd6d56a6a22b)




While the custom_cnn_model was initially defined and trained using the Sequential API, this caused issues when it came to ensemble and chain the model with the ResNet50, which was defined using the Functional API to accommodate the ResNet50's greater complexity. 


# Ensembling Models

<img width="838" alt="Screenshot 2024-09-08 173050" src="https://github.com/user-attachments/assets/94e63c16-57d6-44c1-ae0b-a7ba6ca833c2">
<img width="840" alt="Screenshot 2024-09-08 173153" src="https://github.com/user-attachments/assets/6277ae2c-a9e6-455f-bef1-b31a5d1fff3e">
<img width="844" alt="Screenshot 2024-09-08 173247" src="https://github.com/user-attachments/assets/32f297f2-743d-4a63-9095-dcb8a8e39428">
<img width="875" alt="Screenshot 2024-09-08 173359" src="https://github.com/user-attachments/assets/457282e9-2418-41e3-afaf-6554c509089f">
<img width="854" alt="Screenshot 2024-09-08 173504" src="https://github.com/user-attachments/assets/94c12058-bc1e-420e-a0f3-ed0e73dc0a18">



When CHAINING models, specify data augmentation and rescaling only in first model, not in second
When ENSEMBLING model, either apply data augmentation and rescaling outside of the models or within both models
# Conclusions and Results

## Confusion Matrix of Results for Ensemble Model using Categorical cross entropy loss function

![image](https://github.com/schr0841/cnn_project/blob/main/images/cm_categorical.png)

Above we see the confusion matrix of the ensemble model using the categorical cross entropy loss function as well as the accuracy of 88% on the test set. On the validation set this accuracy increases further to be 97.5%, which is significantly better than each of the constituent pretrained models (84.4%, 75.9%, and 50.6% respectively) or the base CNN model at 29.2%. This is tangible evidence that shows the true benefit of ensembling applied to the unseen validation data. We also showed how using pre-trained models represents a significant boost in model performance over vanilla CNNs, and how this strategy could be fruitful for a wide range of medical imaging problems.



## References

"Chest CT-Scan images Dataset" (2020) Retrieved from https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

"Ensemble model CT scan" (2024) Retrieved from https://www.kaggle.com/code/prthmgoyl/ensemblemodel-ctscan

"How to build CNN in TensorFlow: examples, code and notebooks" (2024) Retrieved from https://cnvrg.io/cnn-tensorflow/

"Convolutional Neural Network (CNN)" (2024) Retrieved from https://www.tensorflow.org/tutorials/images/cnn

"TensorFlow documentation: tf.keras.losses.SparseCategoricalCrossentropy" (2024) Retrieved from https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy

"What does class_mode parameter in Keras image_gen.flow_from_directory() signify?" (Jan. 2020) Retrieved from https://stackoverflow.com/questions/59439128/what-does-class-mode-parameter-in-keras-image-gen-flow-from-directory-signify

"Choosing between Cross Entropy and Sparse Cross Entropy — The Only Guide you Need!" (2023) Retrieved from https://medium.com/@shireenchand/choosing-between-cross-entropy-and-sparse-cross-entropy-the-only-guide-you-need-abea92c84662

"Building an Ensemble Learning Model Using Scikit-learn" (Nov. 18 2018) Retrieved from https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a

"Transfer Learning using EfficientNet PyTorch" (January 17, 2022) https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
