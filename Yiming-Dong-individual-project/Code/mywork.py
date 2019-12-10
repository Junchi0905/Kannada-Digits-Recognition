from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten,BatchNormalization
from keras.optimizers import Adam,Nadam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
validation = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

X=train.drop('label',axis=1)
Y=train.label

Id = test['id']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)
x_train = x_train.values.reshape(-1,28,28,1)/255.0
x_test = x_test.values.reshape(-1,28,28,1)/255.0
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
print(x_train.shape)

imagegen = ImageDataGenerator(
            rotation_range = 20,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.1,
            zoom_range = 0.2
             )
imagegen.fit(x_train)
