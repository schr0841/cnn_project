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


# Ensembling and Chaining Models

We defined, compiled, and trained the ResNet50-based (first_model) and the base CNN model (second_model) individually before ensembling and chaining the two models. We wanted to see if a noticeable improvement in accuracy was possible by combining first_model and second_model over each of the submodels, and whether the method used to combine models (ensembling versus chaining) made a difference in accuracy scores.

To prepare the ResNet50-based model and the base CNN model to be direct ensembled, we defined both first_model and second_model to produce output tensors of identical shape. This required specifying Dense layers for the models' final output layers. We also set the number of units in the final output layers as equal to the class_count value, to reflect the number of output classes in the data (4). We selected the Softmax activation functions for both models because it returns a probability distribution over the class_count classes. This ensured that the shape of the output tensors were (batch_size, class_count) or (None, 4).     

We built the ResNet50-based model, first_model, using the Functional API because it supports more flexibility than the Sequential API in cases of complex model architecture. In particular, the Functional API affords more flexibility when combining pre-trained models with custom layers or sharing layers between models. Since the Functional API allows for explicit definition of the flow of data, it enables fine control over how layers connect and interact. It also supports freezing layers and chaining models.

To eliminate initial errors related to ensembling a Functional API model (first_model) and a Sequential API model (second_model), we rebuilt second_model with the Functional API. The Functional API offered more control over inputs, outputs, and connections, and was better suited to handle the complexities involved in ensembling.


We trained first_model and second_model on the CT chest scan images in the training_set (613 images) and validation_set (72 images) directories. We maintained for evaluation puposes 315 unseen images in the testing_set directory. All images pertained to one of four classes, with one class comprised of cancer-free images and the other three classes indicating three different types of cancer. 

We used the preprocessing.image_dataset_from_directory method to generate the three data sets. This is to say we did not use the ImageDataGenerator to create the datasets, as may have been done elsewhere in this study. Also of note, we set the image_size to (224, 224) for first_model because ResNet50-based models expect images of that size. For purposes of consistency, we set the image_size to (224, 224) for second_model as well. We set label_mode for the three datasets to "int" (integer) because all images in this study belong to one of four classes. 

<img width="914" alt="Screenshot 00a" src="https://github.com/user-attachments/assets/1c553bd2-3721-4a88-a616-a1523fc68a76">
<img width="915" alt="Screenshot 00b" src="https://github.com/user-attachments/assets/a056c7aa-509f-4825-8a81-d6ed52e130a4">


### Building first_model from pretrained ResNet50

The original ResNet50 model is pretrained on over a million images in 1,000 classes in the ImageNet dataset. To make it capable of classifying our chest ct scans into four distinct classes, we had to add some custom layers to it and remove its original output layer. These modifications to the ResNet50 model became our 'first_model'.

To prevent the ResNet50 base_model within first_model from generating a 1,000-classs classification for the ct scans in the input dataset, we "removed" ResNet50's output layer by setting include_top=False. To keep ResNet50's pretrained weights from being updated during training on our data, we froze its layers by specifying layer.trainable = False. Freezing layers enabled the ResNet50 base_model to retain the features it learned from pretraining on the much larger ImageNet data set. In other words, freezing layers prevented the learned features from being overwritten. Common in transfer learning, layer freezing effectively turns a pretrained model into a feature extractor. The custom layers we added to the "feature extractor" then produced the four-class classification, drawing from the ResNet50's learned features.

Specifically, we added the following custom layers:    

a) base_model.output layer to extract the output of the ResNet50 base_model and connect it with the subsequent custom layers. base_model.output represents the features learned by the pretrained ResNet50 model. ResNet50 without its top layer (as we've specified) outputs feature maps instead of classification predictions. The feature maps become the inputs for the subsequent custom layers, which will ultimately result in the classification predictions     

b) BatchNormalization(axis=-1) layer to normalize the ResNet50 base model's output, improving performance and reducing overfitting    

c) Dense(256, activation='relu) layer to learn more complex patterns from the high-level features provided by ResNet50. These more complex patterns will became relevant to the classification task at hand, while the ReLU activation function supported the custom layers to model more intricate relationships between features    

d) Dropout(0.3) to prevent overfitting by forcing the model to learn more robust features and preventing it from becoming too reliant on specific neurons    

e) Dense(class_count, activation = 'softmax') to output a probability distribution across the classes (whose number is given by class_count). Each value in the probability distribution corresponds to the predicted probability that the input image belongs to a given class    

ResNet50, when its top layer is excluded, outputs a feature map with shape (7, 7, 2048). Adding custom layers on top of the ResNet50 base_model allowed us to adapt the pretrained model to fit our specific needs (e.g., completing a four-class classification task, ensembling with the base CNN model, and chaining with the base CNN model). Furthermore, the added BatchNormalization and Dropout layers assisted with regularizing the model, or improving its generalization on unseen data. At the same time, the custom Dense(256) layer reduced the dimensionality of the ResNet50 base_model's output, making it more managable for the final output layer to generate probabilities for each class.


### first_model

<img width="913" alt="Screenshot01" src="https://github.com/user-attachments/assets/f907df78-a421-4290-b85d-287d9dedf82f">
<img width="906" alt="Screenshot02" src="https://github.com/user-attachments/assets/15840ff8-05c8-4569-be6f-304f19b4e251">
<img width="903" alt="Screenshot03" src="https://github.com/user-attachments/assets/88c8107c-3a46-452e-90b9-6844647aeae4">
<img width="913" alt="Screenshot 04" src="https://github.com/user-attachments/assets/c61d8576-6345-4c2a-9c74-a345b50db68f">
<img width="914" alt="Screenshot 05" src="https://github.com/user-attachments/assets/3c702eec-23b1-4138-960f-e9a53b320867">
<img width="914" alt="Screenshot 06" src="https://github.com/user-attachments/assets/7236696d-ceb5-4b35-b3b6-f2716be9ff8c">
<img width="915" alt="Screenshot 07" src="https://github.com/user-attachments/assets/f684737b-7f1c-447f-b14b-645005b2d848">
<img width="909" alt="Screenshot 08" src="https://github.com/user-attachments/assets/9c072431-0181-4b04-bcf9-6a0d845352da">


### Use of Data Augmentation and Rescaling

When ensembling two models, it is appropriate to apply data augmentation and rescaling in both submodels. It is also appropriate to apply data augmentation and rescaling early in the model pipeline. In particular, data augmentation should come before rescaling, right after defining the model's input layer. Because the ResNet50 base_model component of our first_model expected inputs' pixel values to be normalized to a range between 0 and 1, we rescaled the input data before passing it to the ResNet50 layers.  

We applied augmentation and rescaling within first_model and second_model by   
a) defining data augmentation layers within a Sequential model    
b) applying the data augmentation layers to the input tensor    
c) including the augmented inputs as part of a Rescaling layer  

Even though we created first_model with the Functional API, the data augmentation portion of it was created with the Sequential API. Sequential components of larger models, such as data augmentation pipelines, can be integrated into models built with the Functional API. Both APIs produce layers compatibile with the Keras ecosystem. Because data augmentation transforms input data before it reaches the model's core, it doesn't affect the flow of data within the larger model. Defining data augmentation as a Sequential block effectively creates a single layer that acts as any Keras layer. 

Including the augmented inputs as part of the Rescaling layer was necessary because in the Functional API, the data flow between model layers is explicitly defined by passing the output of one layer as the input to next layer. In the scaled_inputs = Rescaling(1./255)(augmented_inputs) statement, the '(augmented_inputs)' explicitly indicates the rescaling operation should be applied to the output of the previous layer, augmented_inputs. Without passing '(augmented_inputs)' as the input, the models would not know which data should be rescaled.   


### second_model

As we designed the base cnn model, second_model, for our four-class classification task, few alternations to this model were necessary until it came time to chain first_model and second_model. The one modification we made prior to training second_model was to re-define and re-train it using the Functional API. In earlier attempts to ensembled second_model with first_model, our initial choice of defining and training second_model using the Sequential API proved complicating. We defined and trained first_model using the Functional API, to accommodate the ResNet50's greater complexity, but not second_model. As such, we neede to redefine, recompiled, and retrained second_model the Functional API. 

<img width="910" alt="Screenshot 09" src="https://github.com/user-attachments/assets/6d4d5ca1-4dce-44ba-98b3-90e6720ad515">
<img width="907" alt="Screenshot 10" src="https://github.com/user-attachments/assets/1bb08068-4152-4ccf-8619-192297c71e0b">
<img width="910" alt="Screenshot 11" src="https://github.com/user-attachments/assets/2789d83c-bd25-4e8d-92b4-5feecfdea36a">
<img width="929" alt="Screenshot 12" src="https://github.com/user-attachments/assets/00a41035-6806-45af-90d8-eb67824362b1">
<img width="928" alt="Screenshot 13" src="https://github.com/user-attachments/assets/261455e6-73b5-4ad2-ba62-0d30652e8929">
<img width="912" alt="Screenshot 14" src="https://github.com/user-attachments/assets/cfdd3fba-05a4-4227-bdbc-624346d344e8">


We compiled both models with the Adam optimizer, and with the loss function set to sparse_categorical_crossentropy. We trained both with x as the training_set dataset and validation_data as the validation_set dataset. Training lasted for 100 epochs, unless our EarlyStopping callback - set to monitor 'val_accuracy' with a patience value of 20 - stopped training early. 

## Ensembling Models

After training first_model and second_model, we created a third model, ensemble_model, to average the first two models' output. The submodels output predictions of shape (None, 4). The ensemble_model takes these outputs as inputs. We do not feed the same image datasets we fed to the submodels to the ensemble_model. Only the predictions (outputs) from first_model and second_model are inputs to ensemble_model.


### Extracting dataset labels

Unlike the original input datasets, which contained images and class labels, the inputs to the ensemble_model lacks labels. Thus, we needed to extract the labels from the TensorFlow datasets and give them to ensemble_model. These labels are necessary for the ensemble_model to compute loss values, which entails comparing predictions to true labels. 

<img width="921" alt="Screenshot 15" src="https://github.com/user-attachments/assets/349ac984-af66-4000-92e4-8acc58df8fd1">
<img width="932" alt="Screenshot 16" src="https://github.com/user-attachments/assets/43efe309-5737-485f-96d4-0992ab5cf365">


### Preparing data and building ensemble model to average outputs

Before we could build the ensemble_model to process the two submodels's output (predictions), we needed to generate predictions from first_model and second_model using the training_set and validation_set. We had used both training_set and validation_set to train each submodel, so we needed predictions from these same models on the same datasets to be inputs to the ensemble_model

Next, we defined the EarlyStopping and ModelCheckpoint callbacks to be used to train ensemble_model. We kept these callback definitions consistent with those used in the two submodels. If the accuracy on the validation dataset did not improve after 20 epochs, the model training would come to an early stop rather than continue on for 100 epochs. Similarly, we defined a filepath to save the best version of the model (that with the maximum validation accuracy). 


<img width="912" alt="Screenshot 17" src="https://github.com/user-attachments/assets/d97b10b3-7c94-4552-833b-b12ca1dc4789">
<img width="907" alt="Screenshot 18" src="https://github.com/user-attachments/assets/9a4d3a77-3bc1-46f4-91d3-c20df954c988">


We defined ensemble_model to average the training_set and validation_set predictions made by first_model and second_model. Keras performs this averaging element-wise across the class probabilities for each sample. Though the submodels' ouput were of shape (None, 4), with None representing variable batch size and 4 representing class probabilities for each image in the batch, Keras implicitly understood that each sample in each batch had a shape of (4,). This was equivalent to the shape ensemble_model expected for its inputs, tensors representing the 4 class probabilities for each sample. We didn't need to reshape any outputs explicitly. The implicit reshaping made it possible to combine the submodel outputs on a sample-by-sample basis rather than processing whole batches of predictions at once. We simply had to specify the shape of each sample with ensemble_input = Input(shape=(4,)).


<img width="913" alt="Screenshot 19" src="https://github.com/user-attachments/assets/ce1519c3-a742-4a86-8092-b41b80949a93">
<img width="907" alt="Screenshot 20" src="https://github.com/user-attachments/assets/99d07b95-cb92-4db1-9591-4dfa66fe95c4">
<img width="914" alt="Screenshot 21" src="https://github.com/user-attachments/assets/9121fc34-0c2a-46dc-8223-34c25be54238">
<img width="913" alt="Screenshot 22" src="https://github.com/user-attachments/assets/5ccd775e-92c9-40eb-be30-30db71dbee08">


We compiled and trained ensemble_model in the same manner as we did its two submodels. Here, our x value became the averaged training_set predictions from the first and second model, while our y values became the true labels corresponding to the averaged training predictions. Finally, the validation_data for the ensemble model became the averaged predictions from the two submodels on the validation dataset and the true labels for the validation dataset itself. 


## Chaining Models

Chaining two models together means creating a composite model, where the first model's output becomes the input for the second model's layers. In this scenario, there is no third model to process the outputs of the two submodels. The output of the first model in a two-model chain is not a classification, but features that will help the second model's layers make a classification. Chaining two models results in a single model that can be trained end-to-end.

Model chaining can be performed using the Functional API in a Keras framework, which allows for flexible connection of layers and models. Unlike in the ensembling, where data augmentation and rescaling can be applied in both submodels, chaining two models requires specifying data augmentation and rescaling only in the first model's layers.


## Modifying first_model from classifier to feature extractor

Because we are turning first_model's output into second_model's input, some adjustments to the original versions of these models became necessary. In particular, first_model needed to be redefined from a classifier to a feature extractor. That is, first_model became tasked with processing raw input data (the ct scans) and producing informative features to be used for classification in second_model's layers. Pretrained models like ResNet50 are often used as feature extractors in transfer learning because they have already learned useful patterns from the large datasets on which they were trained. These patterns, or features, are reusable for new tasks. To diffferentiate first_model and it's modified version, we called our modified first_model 'mod_resnet_model'.

<img width="902" alt="Screenshot 23" src="https://github.com/user-attachments/assets/d4cefbbe-8a3f-4922-a52e-db8b0c760121">
<img width="896" alt="Screenshot 24" src="https://github.com/user-attachments/assets/dd23844d-69f2-4ebc-921b-fe6c621f5a55">
<img width="883" alt="Screenshot 25" src="https://github.com/user-attachments/assets/70de0f23-1638-4e77-a7c1-31d83c48cf2a">


## Modifying second_model to be compatibile for chaining

In order to chain second_model with mod_resnet_model, we needed to omit the data augmentation and rescaling layers. We also needed to remove a number of second_model layers that became redundant when chained with mod_resnet_model. We named this altered version of second_model 'mod_custom_cnn_model' to keep the two distinct. 

The layers we dropped from second_model to create mod_custom_cnn_model were the MaxPooling, Conv2D, and Dropout layers. The first_model layers, based on the ResNet50 model, already performed these operations in the first part of the chain. Reapplying these layers, by including them in second_model's layers, would have been redundant and not necessarily improved performance. The point of chaining first_model and second_model was to use ResNet50 for feature extraction and then use second_model to perform further processing on the extracted features for classification purposes.

Because ResNet50 already included downsampling layers, adding additional pooling and convolution operations with second_model could have resulted in too much downsampling or feature over-processing.

It was appropriate to still include data augmentation and rescaling before the ResNet50 layers since these operations were not part of the feature extraction operations. These pre-processing layers simply prepared the input data for feature extraction.


<img width="901" alt="Screenshot 26" src="https://github.com/user-attachments/assets/3273d338-f2fc-4b9b-9045-2c29fc8df042">
<img width="902" alt="Screenshot 27" src="https://github.com/user-attachments/assets/3f22b4f9-ec67-4976-9ad2-37f58f3a97a7">


## Defining, compiling, and training the chained model

We created the chained model by chaining the modified ResNet50-based model, 'mod_resnet_model', with the modified base cnn model, 'mod_custom_cnn_model'. The two individual models were, themselves, variations of first_model and second_model that made them compatible for chaining. 

By defining mod_resnet_output as mod_resnet_model.output, we specified mod_resnet_model's layers as the first 'link' in the chain. By specifying mod_custom_cnn_output = mod_custom_cnn_model(mod_resnet_output), we passed the first 'link's' output to the second 'link' in the chain, mod_custom_cnn_model, and defined the resulting output as mod_custom_cnn_output. This allowed us to define the composite model, chained_model, as Model(inputs=mod_resnet_model.input, outputs=mod_custom_cnn_output). Before training chained_model, we specified optimizer = Adam(), defined a filepath to save chained_model's best model, and defined equivalent EarlyStopping and ModelCheckpoint callbacks as we'd used previously. We trained chained_model on the dataset training_set and set validation_set as the validation_data.   


<img width="889" alt="Screenshot 28" src="https://github.com/user-attachments/assets/11e7e4dd-414f-43c9-909b-312c2d2a6923">
<img width="906" alt="Screenshot 29" src="https://github.com/user-attachments/assets/72dcca01-284c-48a8-811d-ebb79c3171a4">
<img width="887" alt="Screenshot 30" src="https://github.com/user-attachments/assets/161a3ba4-c89c-48e0-ad00-3bb328fcc6bd">


## Evaluating all four models
When it came to evaluating all four models, first_model and second_model (the submodels) needed to be evaluated on the unseen testing_set dataset to get unbiased performance metrics. 

With first_model, second_model, and chained_model already trained on the training_set and validated on the validation_set, and the best versions of these models saved, we evaluated these three models on the testing_data with the following statements:

first_model_loss, first_model_accuracy = first_model.evaluate(testing_set)
print(f"First model - Loss: {first_model_loss}, Accuracy: {first_model_accuracy}")

second_model_loss, second_model_accuracy = second_model.evaluate(testing_set)
print(f"Second model - Loss: {second_model_loss}, Accuracy: {second_model_accuracy}")

chained_model_loss, chained_model_accuracy = first_model.evaluate(testing_set)
print(f"Chained model - Loss: {chained_model_loss}, Accuracy: {chained_model_accuracy}")

Evaluating the ensemble model was a matter of 
a) averaging the predictions from the two submodels models on the unseen testing_set,  
b) extracting the labels from the testing_set, and   
c) estimating ensemble loss and ensemble accuracy by requesting ensemble_model.evaluate(ensemble_predictions, y_test)


<img width="897" alt="Screenshot 31" src="https://github.com/user-attachments/assets/671b15de-1eb8-48b2-b493-ed34f330a754">
<img width="902" alt="Screenshot 32" src="https://github.com/user-attachments/assets/6fd3212f-6ce8-4f7c-b133-d82cfdb1e2de">


## Table of results

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| first_model  |   |   |  |  |
| second_model   |   |   |  |  |
| ensemble_model  |   |   |  |  |
| chained_model  |   |   |  |  |








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
