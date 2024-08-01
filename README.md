# Medical Image Classification with Convolutional Neural Networks

Project outline: https://docs.google.com/document/d/1iy3c7ik2YjbI88vtGTbysj7AKmAjGtXfu4V3f6WxbHg/edit?usp=sharing


Explore pre-trained models (Xception, VGG19, ResNet50, MobileNet) - not sure useful for this case

Model (sort of) from scratch in tensorflow, get baseline accuracy

Optimization methods: image augmentation, dropout, early stopping



# Overview and Purpose

In this document, we train a convolutional neural network from scratch, and then investigate the added benefit of using pre-trained cnn models on the same dataset. Furthermore, we investigate model ensembling in general and its potential benefits and drawbacks.





From the Kaggle notebook (2) we have some code to work with for pre-trained models. We evaluate each model on test loss, test accuracy, validation loss and validation accuracy. Below are two tables, each populated using a different type of loss function:



## Table of results using CategoricalCrossEntropy loss function and class_mode='categorical' in data generator functions

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| Base cnn  | 9.0188  | 0.1876  | 8.0590 | 0.2917 |
| EfficientNetB3  | 0.0544 | 0.9804 | 0.8051 | 0.8444 |
| ResNet50 | 0.0118 | 0.9951 | 2.0054 | 0.7587 |
| InceptionV3 | 0.0439 | 0.9886 | 2.8829 | 0.5016 |
| Ensemble | 0.3994 | 0.8750 | 0.2605 | 0.9750 |

loss: 9.0188 - accuracy: 0.1876 - val_loss: 8.0590 - val_accuracy: 0.2917

loss: 0.0544 - accuracy: 0.9804 - val_loss: 0.8051 - val_accuracy: 0.8444

loss: 0.0118 - accuracy: 0.9951 - val_loss: 2.0054 - val_accuracy: 0.7587

loss: 0.0439 - accuracy: 0.9886 - val_loss: 2.8829 - val_accuracy: 0.5016

loss: 0.3994 - accuracy: 0.8750 - val_loss: 0.2605 - val_accuracy: 0.9750



## Table of results using SparseCategoricalCrossEntropy loss function and class_mode='sparse' in data generator functions

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| Base cnn  | 0.0112  | 0.9984  | 2.0032 | 0.6111 |
| EfficientNetB3  | 0.0536 | 0.9837 | 0.6506 | 0.8317 |
| ResNet50 | 0.0682 | 0.9886 | 2.0240 | 0.7365 |
| InceptionV3 | 0.0615 | 0.9902 | 2.9775 | 0.5111 |
| Ensemble | 0.9132 | 0.8889 | 0.9594 | 0.8999 |

loss: 0.0112 - accuracy: 0.9984 - val_loss: 2.0032 - val_accuracy: 0.6111

loss: 0.0536 - accuracy: 0.9837 - val_loss: 0.6506 - val_accuracy: 0.8317

loss: 0.0682 - accuracy: 0.9886 - val_loss: 2.0240 - val_accuracy: 0.7365

loss: 0.0615 - accuracy: 0.9902 - val_loss: 2.9775 - val_accuracy: 0.5111

loss: 0.9132 - accuracy: 0.8889 - val_loss: 0.9594 - val_accuracy: 0.8999

## Project topics:

* Investigate what pre-training means
* Sparse categorical vs categorical loss functions / sparse vs non-sparse class mode generators
* Investigate ensemble models
* Ensemble validation set - figure out why only 1 epoch / why validation accuracy so low
* Figure out what .h5 files are doing


## Sparse categorical vs categorical loss functions / sparse vs non-sparse class mode generators

Use this crossentropy loss function when there are two or more label classes. We expect labels to be provided as integers. If you want to provide labels using one-hot representation, please use CategoricalCrossentropy loss. There should be # classes floating point values per feature for y_pred and a single floating point value per feature for y_true.

class_mode: One of "categorical", "binary", "sparse", "input", or None. Default: "categorical". Determines the type of label arrays that are returned: - "categorical" gives 2D output (aka. list of numbers of length N), [0, 0, 1, 0], which is a one-hot encoding (only one number is 1/ "hot") representing the target variable. This is for mutually exclusive labels. A dog cannot be a cat, a human is not a dog. "binary" will be 1D binary labels, "sparse" will be 1D integer labels, - "input" will be images identical to input images (mainly used to work with autoencoders). - If None, no labels are returned (the generator will only yield batches of image data, which is useful to use with model.predict_generator()). Please note that in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly. 

## Investigate ensemble models


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

Ensemble methods are a powerful tool in machine learning, often used in practice to achieve higher performance and more reliable predictions.



## Ensemble validation set - figure out why only 1 epoch / why validation accuracy not showing up

The test_steps parameter in the call to model.evaluate is currently evaluating to 1, so that is why there is only 1 epoch. It seems to be related to the data generator function that we are using - when this parameter was manually increased to 10, a warning was obtained that said the input ran out of data.

The validation accuracy was originally not computed, but we added an appropriate line of code and printed these results to the console. We were then able to add the appropriate result to the table. 


## Confusing Github Error

![error](https://github.com/schr0841/cnn_group_project/blob/main/github_error.png)


## Confusion Matrix of Results for Categorical cross entropy loss function
![image](https://github.com/schr0841/cnn_group_project/blob/main/cm_categorical.png)


## References

"Chest CT-Scan images Dataset" (2020) Retrieved from https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

"Ensemble model CT scan" (2024) Retrieved from https://www.kaggle.com/code/prthmgoyl/ensemblemodel-ctscan

"How to build CNN in TensorFlow: examples, code and notebooks" (2024) Retrieved from https://cnvrg.io/cnn-tensorflow/

"Convolutional Neural Network (CNN)" (2024) Retrieved from https://www.tensorflow.org/tutorials/images/cnn

"TensorFlow documentation: tf.keras.losses.SparseCategoricalCrossentropy" (2024) Retrieved from https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy

"What does class_mode parameter in Keras image_gen.flow_from_directory() signify?" (Jan. 2020) Retrieved from https://stackoverflow.com/questions/59439128/what-does-class-mode-parameter-in-keras-image-gen-flow-from-directory-signify

"Choosing between Cross Entropy and Sparse Cross Entropy — The Only Guide you Need!" (2023) Retrieved from https://medium.com/@shireenchand/choosing-between-cross-entropy-and-sparse-cross-entropy-the-only-guide-you-need-abea92c84662

"Building an Ensemble Learning Model Using Scikit-learn" (Nov. 18 2018) Retrieved from https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a