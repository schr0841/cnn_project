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


# Ensembling versus Chaining Models

We trained the ResNet50-based model (first_model) and the cnn base model (second_model) on the CT chest scan images in the training_set (613 images) and validation_set (315 images) directories. We maintained for evaluation puposes 72 unseen images in the testing_set directory. All images pertained to one of four classes, with one class comprised of cancer-free images and the other three classes pertaining to three different types of cancer. 

The data sets were generated using the tf.keras.preprocessing.image_dataset_from_directory method. This is to say they were not generated using the ImageDataGenerator, as datasets elsewhere in this study may have been generated. Also of note, the image_size was set to (224, 224) because the ResNet50-based model expects images of that size. For purposes of consistency, the image_size was set to (224, 224) for the custom_cnn_model as well. label_mode for the three datasets were set to "int" (integer) because images belong to one of four classes. 

<img width="761" alt="Screenshot 2024-09-11 171824" src="https://github.com/user-attachments/assets/569253a0-4c3b-4329-9596-c99e7dbc291a">
<img width="685" alt="Screenshot 2024-09-11 171939" src="https://github.com/user-attachments/assets/d9ca206c-6903-48b6-a523-ae54cd1d9404">

We defined, compiled, trained, and evaluated both models individually before turning our attention to ensembling and chaining the two models. We wanted to see if a noticeable improvement in accuracy was possible by combining firat_model and second_model, and if there would be a noticeable difference between ensembling accuracy and chaining accuracy.

The original ResNet50 model is pretrained on a multi-million image dataset with 1,000 classes of images. To make it capable of classifying our chest ct scans into four distinct classes, we had to add some custom layers to it, resulting in the definition of first_model. As the cnn base model, second_model, was designed to accomplish the task at hand, alternations to this model were not necessary until it came time to chain the models.

### first_model
<img width="747" alt="Screenshot 2024-09-12 165648" src="https://github.com/user-attachments/assets/5095917b-e89a-4d8f-94be-c6aab6a140fd">
<img width="718" alt="Screenshot 2024-09-12 165800" src="https://github.com/user-attachments/assets/743fc59c-2541-4a0b-bb8b-49d64d33301d">
<img width="769" alt="Screenshot 2024-09-12 165915" src="https://github.com/user-attachments/assets/deaad652-fb39-413b-991b-dae48eb282fd">
<img width="545" alt="Screenshot 2024-09-12 171022" src="https://github.com/user-attachments/assets/9259e7d7-010e-4b8c-9d2a-42681e58c12f">
<img width="686" alt="Screenshot 2024-09-12 171241" src="https://github.com/user-attachments/assets/2bfaba9b-16fb-4cd1-af94-56782297b493">
<img width="670" alt="Screenshot 2024-09-12 171422" src="https://github.com/user-attachments/assets/4b60c346-8567-40c7-b7d7-719b10a53d09">


### second_model

<img width="604" alt="Screenshot 2024-09-12 171903" src="https://github.com/user-attachments/assets/312f8c83-9813-4799-b194-45a155e49f9e">
<img width="532" alt="Screenshot 2024-09-12 172026" src="https://github.com/user-attachments/assets/68ef61bd-6b44-4980-8271-980c9318a537">
<img width="491" alt="Screenshot 2024-09-12 172206" src="https://github.com/user-attachments/assets/373c17e0-9a47-4a5a-8912-804cc7dd4c8b">
<img width="733" alt="Screenshot 2024-09-12 172353" src="https://github.com/user-attachments/assets/8ac5a6ed-8eac-4a55-ada3-335870f3aa3d">
<img width="660" alt="Screenshot 2024-09-12 172528" src="https://github.com/user-attachments/assets/4e076027-c55e-4146-a3f6-368c6cecf27e">


### Use of Data Augmentation and Rescaling

When ensembling two models, it is appropriate to apply data augmentation and rescaling in both submodels. It is also appropriate to apply data augmentation and rescaling early in the model pipeline. In particular, data augmentation should come before rescaling, right after defining the model's input layer. Because the ResNet50 model expects pixel values of the inputs to be normalized to a range between 0 and 1, rescaling needs to be performed before passing the images into ResNet50.  

We applied augmentation and rescaling within first_model (the ResNet50-based model) and second_model (the base cnn model) by   
a) defining data augmentation layers within a Sequential model    
b) applying the data augmentation layers to the input tensor    
c) including the augmented inputs as part of a Rescaling layer  

Even though we created first_model with the Functional API, the data augmentation portion of it was created with the Sequential API. Sequential components of larger models, such as data augmentation pipelines, can be integrated into models built with the Functional API. Both APIs produce layers compatibile with the Keras ecosystem. Because data augmentation transforms input data before it reaches the model's core, it doesn't affect the flow of data within the larger model. Defining data augmentation as a Sequential block effectively creates a single layer that acts as any Keras layer. 

Including the augmented inputs as part of the Rescaling layer was necessary because in the Functional API, the data flow between model layers is explicitly defined by passing the output of one layer as the input to next layer. In the scaled_inputs = Rescaling(1./255)(augmented_inputs) statement, the '(augmented_inputs)' explicitly indicates the rescaling operation should be applied to the output of the previous layer, augmented_inputs. Without passing '(augmented_inputs)' as the input, the models would not know which data should be rescaled.   

### Modifying pretrained ResNet50 model into first_model, compatible with second_model

To prevent the ResNet50-based model from generating a 1,000-classs classification for the ct scans in the input dataset, we "removed" its output layer by setting include_top=False. To keep ResNet50's pretrained weights from being updated during training on our data, we froze its layers by specifying layer.trainable = False. Freezing layers enables the model to retain the features it learned from pretraining on a large image data set. In other words, freezing layers prevents the learned features from being overwritten. Common in transfer learning, layer freezing effectively turns a pretrained model into a feature extractor. The custom layers we added to the "feature extractor" then produced the four-class classification, drawing from the ResNet50's learned features.

Specifically, we added the following layers:    

a) base_model.output layer to extract the output of the ResNet50 and connect it with the subsequent custom layers. base_model.output represents the features learned by the pretrained ResNet50 model. ResNet50 without its top layer (as we've specified) outputs feature maps instead of classification predictions. The feature maps become the inputs for the subsequent custom layers, which will ultimately result in classification predictions     

b) BatchNormalization(axis=-1) layer to normalize the ResNet50 base model's output, improving performance and reducing overfitting    

c) Dense(256, activation='relu) layer to learn more complex patterns from the high-level features provided by ResNet50. The more complex patterns will be relevant to the classification task at hand, while the ReLU activation function supports the cnn to model more intricate relationships between features    

d) Dropout(0.3) to prevent overfitting by forcing the model to learn more robust features and preventing it from becoming too reliant on specific neurons    

e) Dense(class_count, activation = 'softmax') to output a probability distribution across the classes (whose number is given by class_count). Each value in the probability distribution corresponds to the predicted probability that the input image belongs to a given class    

ResNet50, when its top layer is excluded, outputs a feature map with shape (7, 7, 2048). Adding custom layers on top of ResNet50 allows us to adapt the pretrained model to fit our specific needs (e.g., completing a four-class classification task, ensembling with the custom_cnn_model, and chaining with the custom_cnn_model). Furthermore, the added BatchNormalization and Dropout layers assist with regularizing the model, or improving its generalization on unseen data. At the same time, the custom Dense(256) layer reduces the dimensionality of the original ResNet50 model's output, making it more managable for the final output layer which outputs probabilities for each class.

### Modifying second_model for compatibility with first_model

Our second_model also underwent some adjustments to make it compatible with ensembling and chaining with first_model. While second_model was initially defined and trained using the Sequential API, characteristics of this API proved complicating when it came to ensemble and chain with first_model. Our first_model had been defined using the Functional API to accommodate the ResNet50's greater complexity. As such, second_model was redefined, compiled, and trained with the Functional API. 

Both models were compiled with the Adam optimizer, and with the loss function set to sparse_categorical_crossentropy. Both were trained with x as the training_set dataset, with validation_data specified as the validation_set dataset, on 100 epochs, and with an EarlyStopping callback set to monitor 'val_accuracy' with a patience value of 20. 


## Ensembling Models

We created a third model, called ensemble_model, that averages the predictions of first_model and second_model. 

### Extracting labels from training_set, testing_set, and validation_set

The ouputs of first_model and second_model become the inputs to the ensemble model. Unlike the original input data, which contained labels for the images, the new inputs to the ensemble model do not contain labels. Thus, we needed to extract the labels from the TensorFlow datasets and give them to the ensemble model. These labels are necessary for the ensembled model to compute loss values, which entails comparing predictions to true labels. 

<img width="646" alt="Screenshot 2024-09-12 180029" src="https://github.com/user-attachments/assets/ba9baf12-5de0-41f9-8795-7aeebefafeb2">
<img width="882" alt="Screenshot 2024-09-12 175710" src="https://github.com/user-attachments/assets/69059f24-a3d8-4070-923d-044300a42ec1">

### Preparing data and building ensemble model to average outputs

Before we built the ensemble_model to process the two submodels's output (predictions), we needed to generate predictions from first_model and second_model using the training_set and validation_set. Because we used both training_set and validation_set to train the two submodels individually, we needed predictions from both models on these same datasets to serve as the inputs to the ensemble_model.



Next, we defined the EarlyStopping and ModelCheckpoint callbacks to be used to train ensemble_model. We kept these callback definitions consistent with those used in the two submodels. If the accuracy on the validation dataset did not improve after 20 epochs, the model training would come to an early stop rather than continue on for 100 epochs. Similarly, we defined a filepath to save the best version of the model (that with the maximum validation accuracy). 

After that, we defined the training and validation inputs to the ensemble model as the average of the training predictions made by the ResNet50 and the custom_cnn_model and the average of the two submodels' predictions on the validation_set. Because these submodels' outputs/ensemble model's inputs have shape (4,), we set the input shape of the ensemble_input to (4,). 

The model itself is relatively simple since we are only averaging the fist and second models' outputs by dataset. Essentially, this model has only two layers: the input layer and the output layer. We compile and train the ensemble model in the same manner as its two submodels. Here, our x value becomes the averaged training_set predictions from the first and second model, while our y values become the true labels corresponding to the averaged training predictions. Finally, the validation_data for the ensemble model is the averaged predictions from the two submodels on the validation dataset and the true labels for the validation dataset itself. 

<img width="844" alt="Screenshot 2024-09-08 173247t" src="https://github.com/user-attachments/assets/32f297f2-743d-4a63-9095-dcb8a8e39428">
<img width="875" alt="Screenshot 2024-09-08 173359" src="https://github.com/user-attachments/assets/457282e9-2418-41e3-afaf-6554c509089f">
<img width="854" alt="Screenshot 2024-09-08 173504" src="https://github.com/user-attachments/assets/94c12058-bc1e-420e-a0f3-ed0e73dc0a18">

### Chaining Models

Chaining two models together means creating a composite, where the first model's output becomes the second model's input. In this scenario, there is no third model to process the outputs of the two submodels. 

Model chaining can be performed using the Functional API in a Keras framework, which allows for flexible connection of layers and models.

Unlike in the case of ensembling models, where data augmentation and rescaling can be applied in both submodels, chaining two models requires specifying data augmentation and rescaling only in first model.

<img width="802" alt="Screenshot 2024-09-08 200128" src="https://github.com/user-attachments/assets/37e73cb5-bcc5-499d-897c-ad995d6fdd8a">
<img width="879" alt="Screenshot 2024-09-08 200231" src="https://github.com/user-attachments/assets/be08bfe1-f3a5-44b0-b4aa-fa05c2ee3e1e">
<img width="866" alt="Screenshot 2024-09-08 200435" src="https://github.com/user-attachments/assets/3e58715e-ffaf-4d2c-ab69-6391355aa068">


## Modifying first_model and second_model to enable chaining

Because we are turning the output of the first model into the input of the second model, some adjustments to the original versions of first_model and second_model became necessary. In particular, we needed to omit data augmentation and rescaling from second_model and remove a number of its previously-defined layers that become redundant when chaining with the ResNet50-based first_model.

Modifications to the original version of first_model were required to convert this model from a classifier to a feature extractor. That is, the chainable version of first_model needed to process raw input data (the ct scans) and produce a set of informative features that can (eventually) be used for classification tasks. Pretrained models like ResNet50 are often used as feature extractors in transfer learning because they have already learned useful patterns from large datasets on which they were trained. These patterns, or features, are reusable for new tasks. 


<img width="688" alt="Screenshot 2024-09-12 174703" src="https://github.com/user-attachments/assets/903f1534-2125-4450-b4e4-f97746407d0d">
<img width="809" alt="Screenshot 2024-09-12 174943" src="https://github.com/user-attachments/assets/3c474880-558b-45cc-9692-530b47a2a738">
<img width="808" alt="Screenshot 2024-09-12 175058" src="https://github.com/user-attachments/assets/959efbb6-01d3-4880-994f-12b93e64cb99">




## Evaluating all four models

When it comes to evaluating the three models, the first and second models (the submodels) need to be evaluated on the unseen testing_set dataset to get unbiased performance metrics. 

Evaluating the ensemble model is a matter of 
a) averaging the predictions from the first and second models on the unseen testing_set,  
b) extracting the labels from the testing_set, and   
c) Estimating ensemble loss and ensemble accuracy by requesting ensemble_model.evaluate(ensemble_predictions, y_test).

<img width="625" alt="Screenshot 2024-09-08 185804" src="https://github.com/user-attachments/assets/fee59b88-88f0-4418-9eb0-2f0d37882922">
<img width="489" alt="Screenshot 2024-09-08 185910" src="https://github.com/user-attachments/assets/77ae5b95-c029-4897-a964-52654d046a80">

# Validation Loss and Accuracy Metrics for the Three Models

Modified ResNet50 model      Loss: 0.9644      Accuracy: 0.4539

custom_CNN-Model              Loss: 1.3097      Accuracy: 0.4190

Ensemble Model                 Loss: 1.3762      Accuracy: 0.330


## Table of results

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| first_model  | 0.0112  | 0.9644  | 2.0032 | 0.4539 |
| second_model  | 0.0536 | 0.9837 | 0.6506 | 0.8317 |
| ensemble_model | 0.0682 | 0.9886 | 2.0240 | 0.7365 |
| chained_model | 0.0615 | 0.9902 | 2.9775 | 0.5111 |








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
