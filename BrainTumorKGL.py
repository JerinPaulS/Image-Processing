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
filename = '/media/jerinpaul/New Volume/Datasets/Brain MRI KGL'
for sub_folder in os.listdir(os.path.join(filename,'Training')):
    classes.append(sub_folder)
print(classes)

X_train = []
y_train = []
image_size = 160
for i in classes:
    path_train = os.path.join(filename, 'Training', i)
    for j in tqdm(os.listdir(path_train)):
        img = cv2.imread(os.path.join(path_train, j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
    path_test = os.path.join(filename, 'Testing', i)
    for j in tqdm(os.listdir(path_test)):
        img = cv2.imread(os.path.join(path_test, j))
        img = cv2.resize(img, (image_size, image_size))
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

X_train,X_test,y_train,y_test = train_test_split(X_train, y_train, test_size=0.30, random_state=42, stratify=y_train)

labels_train=lb.fit(y_train)
y_train=lb.transform(y_train)
y_test=lb.transform(y_test)

EfficientNet=EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

tf.random.set_seed(45)
model = EfficientNet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.55)(model)
model = tf.keras.layers.Dense(60,activation='elu',kernel_initializer='GlorotNormal')(model)
model = tf.keras.layers.Dropout(rate=0.3)(model)
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=EfficientNet.input, outputs = model)
opt = Adam(learning_rate=0.000016, beta_1=0.91, beta_2=0.9994, epsilon=1e-08)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# summarize the model
print(model.summary())
# fit the model
early_stopping_cb = keras.callbacks.EarlyStopping(patience=9, restore_best_weights=True)

history = model.fit(X_train ,y_train,validation_data = (X_test,y_test), epochs=90, batch_size=13, callbacks=early_stopping_cb)

model.save('/media/jerinpaul/New Volume/Models/BrainMRIKGL3.h5')
model.save_weights('/media/jerinpaul/New Volume/Models/BrainMRIKGL3.h5')

#plot loss and accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.title('Loss and Accuracy', fontsize=18, pad=18)
plt.xlabel('Epochs', labelpad=22, fontsize=14)
plt.ylabel('Percentage', labelpad=22, fontsize=14)
plt.grid(True)
#plt.gca().set_xlim(0,33)
plt.gca().set_ylim(0,5)
plt.savefig('/media/jerinpaul/New Volume/Models/BrainMRIKGLGCA3.png')
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
plt.savefig('/media/jerinpaul/New Volume/Models/BrainMRIKGLHM3.png')
plt.plot()
plt.show()

print(classification_report(y_test,pred,target_names=classes))

'''
        test_size=0.15
accuracy : 97.755 
 loss : 0.062
                  precision    recall  f1-score   support

    glioma_tumor       0.98      0.94      0.96       139
meningioma_tumor       0.95      0.99      0.97       141
        no_tumor       0.97      1.00      0.99        75
 pituitary_tumor       1.00      1.00      1.00       135

        accuracy                           0.98       490
       macro avg       0.98      0.98      0.98       490
    weighted avg       0.98      0.98      0.98       490
'''

'''
        test_size=0.20
accuracy : 96.784 
 loss : 0.116
                  precision    recall  f1-score   support

    glioma_tumor       0.96      0.94      0.95       185
meningioma_tumor       0.96      0.96      0.96       188
        no_tumor       0.94      1.00      0.97       100
 pituitary_tumor       0.99      0.99      0.99       180

        accuracy                           0.97       653
       macro avg       0.97      0.97      0.97       653
    weighted avg       0.97      0.97      0.97       653
'''

'''
        test_size=0.25
accuracy : 95.343 
 loss : 0.141
                  precision    recall  f1-score   support

    glioma_tumor       0.98      0.90      0.93       232
meningioma_tumor       0.92      0.96      0.94       234
        no_tumor       0.94      1.00      0.97       125
 pituitary_tumor       0.98      0.98      0.98       225

        accuracy                           0.95       816
       macro avg       0.95      0.96      0.96       816
    weighted avg       0.95      0.95      0.95       816

'''
'''
        test_size=0.30
accuracy : 93.878 
 loss : 0.184
                  precision    recall  f1-score   support

    glioma_tumor       0.95      0.87      0.91       278
meningioma_tumor       0.89      0.93      0.91       281
        no_tumor       0.94      0.99      0.96       150
 pituitary_tumor       0.97      0.99      0.98       271

        accuracy                           0.94       980
       macro avg       0.94      0.94      0.94       980
    weighted avg       0.94      0.94      0.94       980

'''