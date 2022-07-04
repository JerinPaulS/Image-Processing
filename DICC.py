import numpy as np # matrix tools
import matplotlib.pyplot as plt # for basic plots
import seaborn as sns # for nicer plots
import pandas as pd
from glob import glob
import re
from skimage.io import imread
import os
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam 
from keras.layers import Conv2D, MaxPooling2D

file_loc = '/media/jerinpaul/New Volume/Datasets/DICC'

'''
for dirname, _, filenames in os.walk(file_loc):
    print(dirname,"______")
    for filename in filenames:
        print(os.path.join(dirname, filename))

path= os.path.join(file_loc, 'overview.csv')
overview = pd.read_csv(path, index_col=0)
print(overview.head())

overview['Contrast'] = overview['Contrast'].map(lambda x: 1 if x else 0)

plt.figure(figsize=(10,5))
sns.distplot(overview['Age'])
plt.show()

g = sns.FacetGrid(overview, col="Contrast", height=8)
g = g.map(sns.distplot, "Age")
plt.show()

g = sns.FacetGrid(overview, hue="Contrast", height=6, legend_out=True)
g = g.map(sns.distplot, "Age").add_legend()
plt.show()
'''

os.path.join(file_loc, 'tiff_images', '*.tif')
all_images_list = glob(os.path.join(file_loc, 'tiff_images', '*.tif'))
all_images_list[:5]
#print(all_images_list)

imread(all_images_list[0]).shape
#np.array(np.arange(81)).reshape(9,9)
np.array(np.arange(81)).reshape(9,9)[::3,::3]
np.expand_dims(imread(all_images_list[0])[::4,::4],0).shape
jimread = lambda x: np.expand_dims(imread(x)[::2,::2],0)
test_image = jimread(all_images_list[0])
plt.imshow(test_image[0])
#plt.show()

check_contrast = re.compile(r'/tiff_images/ID_([\d]+)_AGE_[\d]+_CONTRAST_([\d]+)_CT.tif')
label = []
id_list = []
for image in all_images_list:
    id_list.append(check_contrast.findall(image)[0][0])
    label.append(check_contrast.findall(image)[0][1])

label_list = pd.DataFrame(label,id_list)
images = np.stack([jimread(i) for i in all_images_list], 0)

X_train, X_test, y_train, y_test = train_test_split(images, label_list, test_size=0.1, random_state=0)
n_train, depth, width, height = X_train.shape
n_test,_,_,_ = X_test.shape
input_shape = (width,height,depth)

input_train = X_train.reshape((n_train, width,height,depth))
input_train.shape
input_train.astype('float32')
input_train = input_train / np.max(input_train)
input_train.max()
input_test = X_test.reshape(n_test, *input_shape)
input_test.astype('float32')
input_test = input_test / np.max(input_test)

output_train = keras.utils.to_categorical(y_train, 2)
output_test = keras.utils.to_categorical(y_test, 2)
output_train[5]

batch_size = 20
epochs = 40

model2 = Sequential()
model2.add(Conv2D(50, (5, 5), activation='relu', input_shape=input_shape))
model2.add(MaxPooling2D(pool_size=(3, 3))) # 3x3 Maxpooling 
model2.add(Conv2D(30, (4, 4), activation='relu', input_shape=input_shape))
model2.add(MaxPooling2D(pool_size=(2, 2))) # 2x2 Maxpooling 
model2.add(Flatten())
model2.add(Dense(2, activation='softmax'))

model2.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model2.fit(input_train, output_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(input_test, output_test))

score = model2.evaluate(input_test, output_test, verbose=0)
print(score)