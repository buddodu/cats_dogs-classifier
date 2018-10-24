
from keras.models import model_from_json
from keras.models import load_model

json_file=open("model_num1.json","r")
loaded_model1_json=json_file.read()
json_file.close()
loaded_model1=model_from_json(loaded_model1_json)
loaded_model1.load_weights("model_num1.h5")
print("loaded model from disk")


import numpy as np
from keras.preprocessing import image

prediction = image.load_img('/Users/praneethapulivendula/Desktop/MachineLearning projects/cnn/Cats_and_dogs/dataset/single_prediction/dog13.jpg',target_size=(128,128))
prediction=image.img_to_array(prediction)
prediction=np.expand_dims(prediction,axis=0)
result=loaded_model1.predict(prediction)
print(result)
if result==0:
    prediction1='cat'
elif result==1:
   prediction1='dog'
else:
   prediction1="wrong input"
print(prediction1)