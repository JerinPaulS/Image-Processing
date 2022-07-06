import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB2,EfficientNetB3,EfficientNetB5,InceptionResNetV2#,EfficientNetV2S
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import keras
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classes = []
filename = '/media/jerinpaul/New Volume/Datasets/Brain YN'
for sub_folder in os.listdir(os.path.join(filename,'brain_tumor_dataset')):
    classes.append(sub_folder)
print(classes)

X_train = []
y_train = []
image_size = 160
for i in classes:
    path_train = os.path.join(filename, 'brain_tumor_dataset', i)
    for j in tqdm(os.listdir(path_train)):
        img = cv2.imread(os.path.join(path_train, j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
datagen = ImageDataGenerator(
    rotation_range = 7, #rotate images
    width_shift_range = 0.05,
    height_shift_range = 0.05, #shift image in horizontal and vertical
    zoom_range = 0.1, #zoom images
    horizontal_flip = True)

datagen.fit(X_train)
X_train.shape
lb = LabelEncoder()

X_train,X_test,y_train,y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)

labels_train=lb.fit(y_train)
y_train=lb.transform(y_train)
y_test=lb.transform(y_test)

EfficientNet=EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

tf.random.set_seed(45)
model = EfficientNet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.55)(model)
model = tf.keras.layers.Dense(60,activation='tanh',kernel_initializer='GlorotNormal')(model)
model = tf.keras.layers.Dropout(rate=0.3)(model)
model = tf.keras.layers.Dense(2,activation='softmax')(model)
model = tf.keras.models.Model(inputs=EfficientNet.input, outputs = model)
opt = Adam(learning_rate=0.000016, beta_1=0.91, beta_2=0.9994, epsilon=1e-08)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# summarize the model
print(model.summary())
# fit the model
early_stopping_cb = keras.callbacks.EarlyStopping(patience=9, restore_best_weights=True)

history = model.fit(X_train ,y_train,validation_data = (X_test,y_test), epochs=90, batch_size=13, callbacks=early_stopping_cb)

model.save('/media/jerinpaul/New Volume/Models/BrainMRIKGL1.h5')
model.save_weights('/media/jerinpaul/New Volume/Models/BrainMRIKGL1.h5')

#plot loss and accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.title('Loss and Accuracy', fontsize=18, pad=18)
plt.xlabel('Epochs', labelpad=22, fontsize=14)
plt.ylabel('Percentage', labelpad=22, fontsize=14)
plt.grid(True)
#plt.gca().set_xlim(0,33)
plt.gca().set_ylim(0,1)
plt.savefig('/media/jerinpaul/New Volume/Models/BrainMRIYNGCA1.png')
plt.plot()
plt.show()
loss, accuracy = model.evaluate(X_test,y_test)

#print accuracy    
print('Accuracy: %f' % (accuracy*100))

#model=keras.models.load_model('/kaggle/working/EfficientNetB3.h5')  

#model.summary()
loss, accuracy = model.evaluate(X_test,y_test)
model.optimizer.get_config()
print(f'accuracy : {round(accuracy*100,3)} \n loss : {round(loss,3)}')
y_test_labels=lb.inverse_transform(y_test)

#predicted values
pred=np.argmax(model.predict(X_test),axis=1)
pred_labels=lb.inverse_transform(pred)
pd.Series(pred_labels).value_counts()

cm = confusion_matrix(y_test,pred)
fig,ax= plt.subplots(figsize=(10.2,8.1))
a = sns.color_palette("winter_r")  #_r inverts the palette
sns.heatmap(cm, annot=True, fmt='g', linewidths=1, linecolor='white', robust=True, annot_kws={"size":18}, cmap=a)
 #annot_kws: settings about annotations
ax.xaxis.set_ticklabels(classes)
ax.yaxis.set_ticklabels(classes)
plt.yticks(va="center")
plt.title('Confusion Matrix', fontsize=18, pad=18)
plt.xlabel('Actual class', labelpad=22, fontsize=14)
plt.ylabel('Predicted class', labelpad=22, fontsize=14)
plt.savefig('/media/jerinpaul/New Volume/Models/BrainMRIYNLHM1.png')
plt.plot()
plt.show()

print(classification_report(y_test,pred,target_names=classes))

'''
43mins
        test_size=0.15
        ac = elu
accuracy : 96.078 
 loss : 0.16
              precision    recall  f1-score   support

          no       0.95      0.95      0.95        20
         yes       0.97      0.97      0.97        31

    accuracy                           0.96        51
   macro avg       0.96      0.96      0.96        51
weighted avg       0.96      0.96      0.96        51
'''
'''
48mins
        test_size=0.15
        ac = relu
accuracy : 92.157 
 loss : 0.172
              precision    recall  f1-score   support

          no       0.90      0.90      0.90        20
         yes       0.94      0.94      0.94        31

    accuracy                           0.92        51
   macro avg       0.92      0.92      0.92        51
weighted avg       0.92      0.92      0.92        51
'''
'''
        test_size=0.15
        ac = sigmoid
accuracy : 94.118 
 loss : 0.16
              precision    recall  f1-score   support

          no       0.90      0.95      0.93        20
         yes       0.97      0.94      0.95        31

    accuracy                           0.94        51
   macro avg       0.94      0.94      0.94        51
weighted avg       0.94      0.94      0.94        51
'''
'''
        test_size=0.15
        ac = tanh
accuracy : 96.078 
 loss : 0.159
              precision    recall  f1-score   support

          no       0.95      0.95      0.95        20
         yes       0.97      0.97      0.97        31

    accuracy                           0.96        51
   macro avg       0.96      0.96      0.96        51
weighted avg       0.96      0.96      0.96        51
'''