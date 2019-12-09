# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
validation = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

X = train.drop('label', axis=1)
Y = train.label

Id = test['id']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
x_train = x_train.values.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.values.reshape(-1, 28, 28, 1) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(x_train.shape)

imagegen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2
)
imagegen.fit(x_train)

model = Sequential()

model.add(Conv2D(input_shape=(28, 28, 1), filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

fit = model.fit_generator(imagegen.flow(x_train, y_train, batch_size=128), epochs=100,
                          validation_data=(x_test, y_test), verbose=1, steps_per_epoch=100)

loss, accuracy = model.evaluate(x_test, y_test)

plt.figure(figsize=(12, 6))
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.figure(figsize=(12, 6))
plt.plot(fit.history['accuracy'])
plt.plot(fit.history['val_accuracy'])
plt.title('model train vs validation acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

test = test.drop(['id'], axis=1)
test = test.values.reshape(test.shape[0], 28, 28, 1) / 255.0
predict = model.predict_classes(test)

y_pre = model.predict(x_test)
y_pre = np.argmax(y_pre, axis=1)
y_test = np.argmax(y_test, axis=1)

result = confusion_matrix(y_test, y_pre)
result = pd.DataFrame(result, index=range(0, 10), columns=range(0, 10))
print(result)

x = (y_pre - y_test != 0).tolist()
x = [i for i, l in enumerate(x) if l != False]

fig, ax = plt.subplots(1, 4, sharey=False, figsize=(15, 15))

for i in range(4):
    ax[i].imshow(x_test[x[i]][:, :, 0])
    ax[i].set_xlabel('Real {}, Predicted {}'.format(y_test[x[i]], y_pre[x[i]]))

submission = pd.DataFrame({'id': Id,
                           'label': predict})
submission.to_csv(path_or_buf="submission.csv", index=False)
submission.head()

print('test loss', loss)
print('test accuracy', accuracy)