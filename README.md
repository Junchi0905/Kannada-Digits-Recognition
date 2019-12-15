# Kannada Digits Recognition
This is a repository for Machine Learning II final project: Kannada Digits Recognition

Team Members: Junchi Tian, Jingshu Song, Yiming Dong

# Introduction

For this final project, we used a recently released dataset of Kannada digits. Kannada is the official and administrative language of the state of Karnataka in India with nearly 60 million speakers worldwide. The language has roughly 45 million native speakers and is written using the Kannada script. Distinct glyphs are used to represent the numerals 0-9 in the language that appear distinct from the modern Hindu-Arabic numerals in vogue in much of the world today. The following picture shows the figure of Kannada.

![image](https://github.com/Junchi0905/Kannada-Digits-Recognition/blob/master/kannada.jpg)

# Description of dataset

This dataset is from Kaggle (https://www.kaggle.com/c/Kannada-MNIST/overview).

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine, in the Kannada script. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive. The training data set has 785 columns. The first column, called label, is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image. The test data set is the same as the training set, except that it does not contain the label column.

