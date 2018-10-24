#image preprocessing
#part 1 ->convolutional neural netowrk
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import Callback
from keras import backend
import os
from keras.preprocessing.image import ImageDataGenerator


 
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
 
script_dir = os.path.dirname('/Users/praneethapulivendula/Downloads/Convolutional_Neural_Networks')
training_set_path = os.path.join(script_dir, '/Users/praneethapulivendula/Downloads/Convolutional_Neural_Networks/dataset/training_set')
test_set_path = os.path.join(script_dir, '/Users/praneethapulivendula/Downloads/Convolutional_Neural_Networks/dataset/test_set')
 

#initialize cnn
classifier=Sequential()

#convolutional layer
classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))

#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening
classifier.add(Flatten())

#cnn connection
classifier.add(Dense(activation='relu',units=64))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation='sigmoid',units=1))

#compiling the cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting cnn to images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/Users/praneethapulivendula/Downloads/Convolutional_Neural_Networks/dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/Users/praneethapulivendula/Downloads/Convolutional_Neural_Networks/dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

history=LossHistory()

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/32,
                         epochs=90,
                         validation_data=test_set,
                         validation_steps=2000/32,
                         workers=12,
                         max_q_size=100,
                         callbacks=[history])



import numpy as np
from keras.preprocessing import image

prediction = image.load_img('/Users/praneethapulivendula/Downloads/Convolutional_Neural_Networks/dataset/single_prediction/dog1.jpeg',target_size=(128,128))
prediction=image.img_to_array(prediction)
prediction=np.expand_dims(prediction,axis=0)
result=classifier.predict(prediction)
training_set.class_indices
print(result)
if result==0:
    prediction1='cat'
else:
   prediction1='dog'
print(prediction1)



from keras.models import model_from_json
from keras.models import load_model

model_json=classifier.to_json()

with open("model_num1.json","w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model_num1.h5")




























