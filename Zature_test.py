from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('zature_model.h5') #Modeli yüklüyoruz
img=image.load_img('chest_xray/val/NORMAL/NORMAL2-IM-1437-0001.jpeg',target_size=(224,224))
imagee=image.img_to_array(img) #X-Ray verisini pixellere dönüştürüyoruz
imagee=np.expand_dims(imagee, axis=0)
img_data=preprocess_input(imagee)
prediction=model.predict(img_data)
if prediction[0][0]>prediction[0][1]:  #Modelin tahminlemesini yazıyoruz.
    print('Kişi sağlıklıdır.')
else:
    print('Kişide zatüre vardır.')
print(f'Tahminler: {prediction}')
