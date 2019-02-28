# Readme
This is my work from Udacity Data Scientist Nanodegree project about Deep Learning.
The goal is to use transfer learning to build and train an image classifier which can recognize different species of flowers.
Data used here to train/valid/test the model can be found [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

My approach is to use the pretrained vgg19 model, and replace the last part with a classifier that can output 102 class labels.
First part of the project, the model is trained in Jupyter Notebook, after 10 epochs of training, the validation accuracy reached 92%, so I tested it against the test dataset covering all categories, with accuracy reaching 90.2%.
As the second part, the model is then implemented with stand alone Python scripts, which can be run using command line interface.
Using `train.py`, there are two model architectures as options: vgg19 and resnet34. We can train the model with new datasets, and change the hyperparameters such as learning rate, hidden layers, moreover, the model training process can be saved and reloaded afterwards, if the accuracy is not satisfying.
Using `predict.py`, the trained model can then be used to predict the class for one input image, and output the top `k` predictions and corresponding probabilities.
