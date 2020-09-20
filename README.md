# Dog Classifier 

Dog breed classifier built for an introductory machine learning project at UNSW. Using a dataset of 10 thousand labelled images, we trained the model to recognise 120 distinct dog breeds.

**Since the paper written on the project cannot be published, an abridged version can be found below**

---

Dataset: https://www.kaggle.com/c/dog-breed-identification/overview

## Implementation 

Our final implementation was coded in Python 3.8, implementing TensorFlow 2.2 and Keras as the wrapper. We also use sk-learn. Like in our later experiments, we chose VGG16 as a pre-trained neural network to discern high level features from the image, saving a significant amount of compute power required to train a similar model to discern said details from an image.

Using the pre-trained VGG16 model on the entire dataset, we are able to extract key, higher-level features from the image that will be useful in drawing implications about specific dog breeds when we generate our model. We then split these key features into a “test train split” function, so that we have training and testing data.

We then used a RandomForestClassifier, which contains a large set of decision trees that work on various subsets of the data. The results from these decision trees are averaged to improve accuracy and reduce potential over-fitting to the data. In our implementation, we chose 500 decision trees.

We fit our training data to our RandomForestClassifier. In this process, 500 decision trees are randomly created and altered to the training data as it receives feedback from the labelled training set. Each tree takes a sub-sample of the data and is optimised based on feedback from each prediction it makes.

## Results 

After our final implementation, our model achieved a 75% accuracy on the test split of our data. By breed (label), there was a variation in accuracy; this is likely due to shared visual features between these breeds as the accuracy does not correlate with the number of images present for each breed. We can see the accuracy per breed in the confusion matrix below; a straight line indicates accurate predictions, and a deeper blue indicates a higher number of correct predictions. As we can see, the model generated is relatively accurate and does not have any specific breed that is unusually inaccurate (an outlier).

## Conclusion

By utilising a pre-trained convolutional neural network to discern higher level features, and then a forest of decision trees fit to the higher level features and the correct training labels, we were able to distinguish between 120 different dog breeds with an average of 75% accuracy on our unseen validation set. Due to its accuracy, this model is unlikely to be precise enough for critical real-world applications, such as use in veterinary clinics. The model could - however - be used in non-critical consumer-facing software for breed identification.

To further improve our model, we would require a larger dataset with more examples per label (dog breed) to further train our model on. Additionally, access to more compute power would enable us to train - or modify - a larger set of decision trees to help further make sense of the higher-level features provided by the neural network. Future implementations may use neural networks to make the predictions instead of using decision trees; however, this approach takes a significantly larger dataset and more compute power.
