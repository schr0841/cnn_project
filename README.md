# Medical Image Classification with Convolutional Neural Networks

## Executive Summary

We defined, compiled, and trained two CNN submodels - one custom and one pre-trained - individually before ensembling and chaining them. We looked for a noticeable improvement in accuracy by combining the submodels over each of the submodels, individually.    

The data for this project was obtained here: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images  
  
## Overview and Purpose
  
Can we train a custom convolutional neural network (CNN) model from scratch to classify CT chest scan images as indicating one of the following four categories (classes): Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, or normal cells? How well does the model perform?
  
What happens if we add to our model a pre-trained CNN model by employing transfer learning and model ensembling? Will we see improved accuracy scores with either of these methods?
  
In our project, we compared the accuracy values obtained with our dataset of chest CT images in the following four model scenarios:  
a) the pre-trained ResNet50 model (model_one)   
b) our custom CNN (model_two)  
c) ensembling the output of model_one and model_two (model_three)  
d) chaining model_one and model_two into model_four (transfer learning)  

Concepts:
Convolutional Neural Networks  
Pretained models  
Model Ensembling  
Transfer Learning/model chaining  
  
  
## Convolutional Neural Networks  
  
CNNs use convolutional and pooling layers to automatically and hierarchically learn features from images, and use fully connected layers to classify those features into predefined categories. This process enables CNNs to effectively handle and classify complex visual data.  
    
We built our custom CNN (model_two) and our ResNet50-based pre-trained model with the following components:  
  
1. Input Layer: Our input images are represented as matrices of pixel values.   

2. Convolutional Layers: These layers applied convolutional filters (or kernels) to the inputs. Each filter scanned the images and performed a convolution operation involving element-wise multiplication and results summing. These layers extracted features like edges, textures, and patterns from each image and produced a feature map highlighting the presence of specific features in different parts of the image.  
  
3. Activation Function: We applied activation function ReLU (Rectified Linear Unit) to introduce non-linearity into the model, which helped the network learn more complex patterns.  
  
4. Pooling Layers: We used max pooling to reduce the spatial dimensions of the feature maps by taking the maximum value from a subset of the feature map. This reduced the number of parameters and computations, helping the network become more robust to variations in image.  
  
5. Flattening (model_two only): Because the output from the convolutional and pooling layers was a multi-dimensional tensor, we needed to flatten the tensor to a one-dimensional vector before feeding it into the fully connected layers.    
  
6. Fully Connected Layers: Similar to traditional neural networks, where each neuron is connected to every neuron in the previous layer, the fully connected layers combined the features learned by the convolutional and pooling layers to make a final prediction.  
  
7. Output Layer: We chose a softmax function capable of outputting probabilities for each of the four classes, indicating the network's prediction of Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, or normal cells.   
    
 
## Pretrained Models

Pre-training a neural network involves training a model on a large, broad, general-purpose dataset before fine-tuning it on a specific task (a new set of specific, likely previously unseen data). The ResNet50 model is a well-known model that was trained on the ImageNet database, a collection of millions of images classified across thousands of categories.   

During pre-training, the model learns to identify and extract general features from the input data, such as images' edges, textures, and shapes. These features become broadly useful across new tasks and data domains, even if the new data was never part of the training data.

The benefits of pre-training include improved performance, better generalization, and reduced training time. Pre-training allows the model to leverage knowledge learned from a large and diverse dataset. This accumulated knowledge can lead to better performance on the new task, especially when the new dataset is small or lacks diversity. Training a model from scratch can be computationally expensive and time-consuming. Pre-training on a large dataset and then fine-tuning it can significantly reduce the time required to achieve good performance.

Pre-trained models often generalize better to new tasks because they start with a solid understanding of basic features and patterns, which can help improve accuracy on the new task. Pre-training can be a powerful technique, especially when data are scarce or where training a model from scratch would be impractical given resource constraints.  

## Model adjustments and considerations for model compatibility
  
To make our custom CNN and the pre-trained CNN compatible with each other for direct ensembling and transfer learning puposes, we needed to make adjustments to the original ResNet50 model, and specify our custom CNN carefully. Note that we referred to our 'adjusted' ResNet50 model as our ResNet50-based model.   
  
For our pre-trained model, we built a ResNet50-based model called first_model. It's base (base_model) was the original ResNet50 model with weights reflecting the ImageNet database of images on which it was trained. To this base we made the following modifications:
  * Specified the img_size, channels, img_shape, and class_count to be identical's to those in the custom CNN
  * Defined the same data augmentation layers as in our custom CNN
  * Applied data augmentation to the input tensor
  * Applied the same rescaling defined in our custom CNN
  * Specified the input tensor as the scaled inputs
  * Avoided outputting the 1,000-class predictions for which ResNet50 was originally trained by removing its top layer
  * Avoided re-training ResNet50s pre-trained knowledge by making the base_model's layers untrainable
  * Added custom layers to the base_model to produce first_model, which was capable of outputting predictions as similarly to second_model as possible, including
    *  BatchNormalization layer
    *  0.3 Dropout layer
    *  Dense output layer capable of producing predictions for a four-class problem
  
To prevent errors related to ensembling a Functional API model and a Sequential API model, we built both first_model and second_model with the Functional API. The Functional API offered more control over inputs, outputs, and connections, and was better suited to handle the complexities involved in model ensembling than the Sequential. The Functional API supported more flexibility in complex model architecture than the Sequential API, including combining pre-trained models with custom layers. Because Functional API allowed data flow to be explicitly defined, it supported freezing layers and chaining models.

     
## The two CNN sub-models 
  
### first_model, the ResNet50-based model  
from tensorflow.keras.layers import BatchNormalization  
  
img_size = (224, 224)                               # 224x224 what ResNet50 expects  
channels = 3                                        # one channel each for Red, Blue, Green (color images)  
img_shape = (img_size[0], img_size[1], channels)  
class_count = len(training_set.class_names)         # class_names auto defined when image_dataset_from_directory creates dataset  
  
inputs = Input(shape=(224, 224, 3))                 # define input tensor
  
data_augmentation = tf.keras.Sequential([RandomFlip("horizontal", RandomRotation(0.2), RandomZoom(0.2)])  # define data augmentation layers directly from tf.keras.layers  
  
augmented_inputs = data_augmentation(inputs)  #apply augmentation to input tensor, store results in 'augmented_inputs'
  
scaled_inputs = Rescaling(1./255)(augmented_inputs)  # normalize pixel values; explicitly indicate rescaling to previous layer's inputs  
  
base_model = ResNet50(       # define ResNet50-based model as base model for first_model  
    weights='imagenet',      # use the weights resulting from training on ImageNet database
    include_top=False,       # ignore top layer of original ResNet50; we'll define new top layer with only 4 classes
    input_tensor=scaled_inputs,   # instantiate with scaled_inputs as input tensor  
    pooling='max')                # base model outputs tensor compatible with Dense layers, no need to flatten tensor  

for layer in base_model.layers:   
    layer.trainable = False  # freeze ResNet50 layers to prevent re-training pre-training    
  
x = base_model.output  # base_model's output to get custom layers on top of it
x = BatchNormalization(axis=-1)(x)    # including BatchNormalization before Dense can yield better training and performance  
x = Dense(256, activation='relu')(x)  
x = Dropout(0.3)(x)  
  
outputs = Dense(class_count, activation='softmax')(x)  # define layer, output vector with 4 classes to allow ensembling 

first_model = Model(   # because outputs variable represents model final output, 
inputs=inputs,         # when defining model using Model class,
outputs=outputs        # use outputs = outputs
)    
  
  
### second_model, custom CNN  
from tensorflow.keras.layers import Input, RandomFlip, RandomRotation, RandomZoom, Dense  
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Dropout, Flatten  
from tensorflow.keras.models import Model  
  
img_size = (224, 224)       # resize to 224x224, what ResNet50 expects -- needed for chaining models    
channels = 3                # one channel each for Red, Blue, Green (color images)   
img_shape = (img_size[0], img_size[1], channels)     
class_count = len(training_set.class_names)  # class_names auto defined when image_dataset_from_directory creates dataset    
input_tensor = Input(shape=img_shape)        # define input layer
    
x = RandomFlip("horizontal")(input_tensor)   # apply 3 data augmentation layers  
x = RandomRotation(0.2)(x)  
x = RandomZoom(0.2)(x)  
x = Rescaling(1./255)(x)                     #apply rescalling
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)  
x = MaxPooling2D(pool_size=(2, 2))(x)  
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)  
x = MaxPooling2D(pool_size=(2, 2))(x)  
x = Dropout(0.25)(x)  
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)  
x = MaxPooling2D(pool_size=(2, 2))(x)  
x = Dropout(0.25)(x)  
x = Flatten()(x)  
x = Dense(128, activation='relu')(x)  
x = Dropout(0.25)(x)  
  
output_tensor = Dense(class_count, activation='softmax')(x)  # define output layer for 4-class problem
second_model = Model(inputs=input_tensor, outputs=output_tensor)  # define model 

  
## Model Ensembling  
  
## Ensembling models

Ensembling models entails combining the individual predictions of multiple models on the same dataset to try to make better predictions on that dataset. Ensemble models can improve upon the predictive performance of individual models. The idea behind ensembling models is that if different models make different types of errors, we may reduce the overall error rate by combining their predictions. 

In this project, we chose to combine our two submodels' predictions in an ensemble model, model_three, by averaging their individual output. Here, each model contributing to model_three is weighted equally in the ensemmble model. It is possible, however, to configure a weighted average ensemble in which better-performing submodels contribute more to the ensemble than poorer-performing submodels. 

There are additional techniques for combining submodel predictions. In bootstrap aggregating, multiple models are trained on different subsets of the same training data and then ensembled. Boosting models occurs when models are trained sequentially, allowing later models to correct the errors made by earlier models. The voting technique makes a final prediction by taking a majority vote of the predictions made by the various submodels. 

Ensemble models can yield improved accuracy over their individual submodels by reducing overfitting. They may exhibit more robustness to changes in input data than their submodels. On the other hand, ensemble models can entail increased complexity, reduced ease of interpretability, and greater computational costs than their submodels individually.    

    
### Ensembling first_model and second_model: ensemble_model

First, we defined the full file paths to our best saved models for first_model and second_model, and loaded them from saved. keras.

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Average
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization

first_filepath = os.path.join(base_dir, 'first_model.keras')     
second_filepath = os.path.join(base_dir, 'second_model.keras')   
first_model = load_model(first_filepath)                         
second_model = load_model(second_filepath)

Second, we extracted labels from the TensorFlow datasets (training_set, testing_set, validation_set) we had created before building first_model and second_model, using the tf.keras.preprocessing.image_dataset_from_directory method. Ensemble models need labels in order to compute loss (by comparing predictions to true lables) and update models during training. 

Because training_set and validation_set are tf.data.Dataset objects that return batches of (images, labels), we could loop through the datasets to extract images and lables and combine batch-wise labels in single tensors. 

def get_labels(dataset):
    labels = []
    for _, batch_labels in dataset:          # loop iterates over dataset, where batch_labels contains labels for batch of images
                                             # _ ignores image data; we're only interested in labels
        labels.append(batch_labels.numpy())  # convert TensorFlow tensors (which hold labels) into NumPy arrays
                                             # convert label arrays for each batch appended to labels list
    return np.concatenate(labels, axis=0)    # after iterating through all batches, merge all label arrays from list into
                                             # single NumPy array, resulting in single array containing all labels from dataset
y_train = get_labels(training_set)           # y_train will contain all extracted labels from training_set
y_test = get_labels(testing_set)             # y_test will contain all extracted labels from testing_set
y_val = get_labels(validation_set)           # y_val will contain all extacted labels from validation_set
  
Third, we generated submodel predictions for the training and validation datasets. 
    
preds_first_model_train = first_model.predict(training_set)
preds_second_model_train = second_model.predict(training_set)
preds_first_model_val = first_model.predict(validation_set)
preds_second_model_val = second_model.predict(validation_set)

Fourth, we define EarlyStopping and ModelCheckpoint callbacks and a file to save the ensemble_model. 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
ensemble_filepath = os.path.join(base_dir, 'ensemble_model.keras')     # define new file path to save ensemble_model
checkpoint = ModelCheckpoint(ensemble_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')   # save max best ensemble_model

Fifth, we built and trained the ensemble_model to process the combined predictions. The ensemble model's input shape reflected how outputs would get combined, two predictions per class, with an output shape of (None, 4). 

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Average, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    
ensemble_input_train = (preds_first_model_train + preds_second_model_train) / 2     # average both models' predictions for 'training set'
ensemble_input_val = (preds_first_model_val + preds_second_model_val) / 2           # average both models' predictions for 'validation set'
  
ensemble_input = Input(shape=(4,))              # build ensemble_model, define input layer with shape corresponding to four classe
final_output = Dense(4, activation='softmax')(ensemble_input)     # add dense layer, suited for 4-class problem
ensemble_model = Model(inputs=ensemble_input, outputs=final_output)   # define ensemble model
  
ensemble_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   # compile ensemble model
  
history = ensemble_model.fit(                           # Keras fit method trains model for fixed number of epochs using provided training data and labels,
                                                        # returns history object with loss & accuracy values at each epoch
    x=ensemble_input_train,                             # specifies input data as averaged predictions from submodels
    y=y_train,                                          # use labels from original dataset to specify training data labels; y_train represents true labels
                                                        # corresponding to ensemble_input_train predictions
    validation_data=(ensemble_input_val, y_val),        # specify validation data to be used to evaluate model after each epoch: ensemble_input_val contains  
                                                        # averaged predictions from submodels on validation set; y_val contains true labels
    epochs=100,                                         # model trains for 100 epochs, updating weights after each batch of data within epoch
    callbacks=[early_stopping, checkpoint],             # stop training early if val loss or accuracy doesn't improve, helps prevent overfitting;
                                                        # checkpoint saves model’s weights at certain points to ensure restoration of best model
    verbose=1                                           # detailed progress of each epoch: loss, accuracy, validation metrics displayed in output
)










# 1. Average Predictions:
    # a. Average predictions from both models for 'training set'
ensemble_input_train = (preds_first_model_train + preds_second_model_train) / 2

    # b. Average predictions from both models for 'validation set'
ensemble_input_val = (preds_first_model_val + preds_second_model_val) / 2

# 2. Build ensemble_model
  # a. Define input layer for ensemble_model (shape corresponds to 4 classes)
ensemble_input = Input(shape=(4,))

  # b. Add dense layer
final_output = Dense(4, activation='softmax')(ensemble_input)  # 4 classes

  # c. Define ensemble model
ensemble_model = Model(inputs=ensemble_input, outputs=final_output)

# 3. Compile ensemble model
ensemble_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# 4. Train ensemble model on averaged predictions

history = ensemble_model.fit(                           #initiates training for ensemble_model; Keras fit method trains model for fixed number of epochs using
                                                        #provided training data and labels. Returns history object with loss & accuracy values at each epoch

    x=ensemble_input_train,                             #specifies input data to train ensemble_model; averaged predictions from submodels with shape (None, 4)

    y=y_train,                                          #Use same labels from original dataset to specify labels for training data; y_train represents true labels
                                                        #corresponding to ensemble_input_train predictions; labels required to calculate loss during training

    validation_data=(ensemble_input_val, y_val),        #specifies validation data to be used to evaluate model after each epoch
                                                        #ensemble_input_val contains averaged predictions from submodels on validation set; y_val contains true labels

    epochs=100,                                         #Specifies # of epochs for which model will train; epoch = one complete pass through entire training dataset
                                                        #Model will train for 100 epochs, updating weights after each batch of data within an epoch

    callbacks=[early_stopping, checkpoint],             #early_stopping stops training early if val loss or accuracy doesn't improve for specified # of epochs, helps
                                                        #prevent overfitting; checkpoint saves model’s weights at certain points to ensure restoration of best model

    verbose=1                                           #detailed progress of each epoch, loss, accuracy, validation metrics displayed in output
) 


  
## Transfer Learning   

After pre-training, the model is applied to a new, specific dataset and classification task in a process known as transfer learning. The pre-trained model's weights, optimized during pre-training, become the starting point for training on a new, often smaller, dataset. The model learns the specifics of the new task while leveraging the general features it learned during pre-training. In our project, the smaller dataset consisted of the CT-Scan images with different types of chest cancer versus normal cells. The ResNet50 model (pre_trained

The process of combining a pre-trained model with a custom CNN is called transfer learning.   













## Table of results using CategoricalCrossEntropy loss function and class_mode='categorical' in data generator functions

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| Base cnn  | 9.0188  | 0.1876  | 8.0590 | 0.2917 |
| EfficientNetB3  | 0.0544 | 0.9804 | 0.8051 | 0.8444 |
| ResNet50 | 0.0118 | 0.9951 | 2.0054 | 0.7587 |
| InceptionV3 | 0.0439 | 0.9886 | 2.8829 | 0.5016 |
| Ensemble | 0.3994 | 0.8750 | 0.2605 | 0.9750 |












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


<img width="889" alt="Screenshot 28" src="https://github.com/user-attachments/assets/4a27f82a-462b-4287-af38-e2984f576bf3">
<img width="913" alt="Screenshot 29" src="https://github.com/user-attachments/assets/78e3e396-7543-421a-a851-cb66eb0a264f">
<img width="886" alt="Screenshot 30" src="https://github.com/user-attachments/assets/6c06d877-6cad-420b-9d57-806e501c9b28">


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

| model          |   train_loss  | train_accuracy |   val_loss   | val_accuracy |   test_loss   |  test_accuracy  |
|----------------|---------------|----------------|--------------|--------------|---------------|-----------------|
| first_model    | 0.6271 | 0.7406 | 1.009 | 0.6250 | 1.0288 | 0.5301 | 
| second_model   | 0.4488 | 0.8189 | 0.6815 |  0.8194 | 2.9589 | 0.3746 |
| ensemble_model | 1.4364 | 0.2120 | 1.4026 | 0.2777 | 1.4087 |  0.2444 |
| chained_model  | 0.8707 | 0.6215 | 0.9116 | 0.5833 | 1.0094 |  0.5142 |

We noted some unexpected results when combining the two models. Neither the ensemble_model nor the chained_model outperformed the first_model (the ResNet50-based classifier). The accuracy results for the ensemble_model were especially low, when compared with the two submodels.

It is unusual for an ensemble model that combines its submodels' output to have lower accuracy than its individual submodels. Such results can indicate that there's an issue with prediction averaging, if the models' outputs are raw logits or probabilities. Because the two classification models are generating averageable probabilities, however, averaging errors are not at play.

Likewise, we can rule out the possibility that the model was trained using pseudo-lables rather than true labels, since we explicitly specified the relevant true lables as the validation_data. Pseudo-labeling would have occured if we trained the ensemble model on the submodel predictions as labels, instead of using the true labels.

Finally, a problem with the evaluation methodology doesn't explain the lower accuracy scores for the ensemble_model. The evaluation of the ensemble model wass consistent with how the model was trained (e.g., on averaged predictions). 

It could be that the two submodels are underperforming or have biases. If this is the case, averaging the two submodels' predictions would not necessarily improve performance. It's also possible that averaging the submodels' predictions is exacerbating weaknesses in the two models if the models are making similar errors. 

Overfitting or underfitting could also be a factor. Averaging the predictions of models that overfit the training data could result in poor generalization to unseen data. Averaging predictions could also be problematic if the submodels are underfitting the training data because the averages could be failing to capture complex patterns. 

Alternatively, it might be the case that simple averaging is not appropriate when ensembling our two models. Averaging predictions when one model submodel is significantly better than the other can dilute the effectiveness of the stronger model. Using weighted averaging instead of simple averaging might be called for in this case, as first_model has significantly better accuracy scores than second_model. A possible next step could be ensembling first_model and second_model with weighted averages. 


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
