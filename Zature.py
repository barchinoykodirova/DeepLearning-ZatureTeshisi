from keras.models import Model
from keras.layers import Flatten,Dense
from keras.applications.vgg16 import VGG16  #Gerekli modüller yükleniyor
import matplotlib.pyplot as plot
from glob import glob

IMAGESHAPE = [224, 224, 3] #Görüntü boyutunu 224 x 224 olarak belirtiyoruz; VGG16 mimarisi için sabit bir boyuttur
                           #3 parametresini RGB tipi görsellerle çalıştığımız gösteriyor.
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

#modeli oluşturacağımız datasetimiz..
training_data = 'chest_xray/train'
testing_data = 'chest_xray/test' #Eğitme ve test veri seti bulunduran dizinlerimiz

for each_layer in vgg_model.layers:
    each_layer.trainable = False #Tüm katmanları eğitmemek için eğitilebilir ayarını false olarak ayarlayıruz,.
classes = glob('chest_xray/train/*') #Eğitim veri setimizde kaç tane class bulunduğunu buluyoruz
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)
final_model = Model(inputs=vgg_model.input, outputs=prediction) #VGG çıktısını ve tahminini birleştiriyoruz, bunların hepsi birlikte bir model oluşturacak.
final_model.summary() #Özeti gösteriyoruz
final_model.compile( #Adam optimizer ve optimizasyon metriğini doğruluk olarak kullanarak modelimizi derliyoruz.
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, #veri setimizi keras'a, ImageDataGenerator kullanarak import ediyoruz.
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
testing_datagen = ImageDataGenerator(rescale =1. / 255)
training_set = train_datagen.flow_from_directory('chest_xray/train', #imajları ekleme.
                                                 target_size = (224, 224),
                                                 batch_size = 4,
                                                 class_mode = 'categorical')
test_set = testing_datagen.flow_from_directory('chest_xray/test',
                                               target_size = (224, 224),
                                               batch_size = 4,
                                               class_mode = 'categorical')
fitted_model = final_model.fit( #Modeli fit etme.
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
plot.plot(fitted_model.history['loss'], label='training loss') #Doğrulukları çiziyoruz
plot.plot(fitted_model.history['val_loss'], label='validation loss')
plot.legend()
plot.show()
plot.savefig('LossVal_loss')
plot.plot(fitted_model.history['accuracy'], label='training accuracy')
plot.plot(fitted_model.history['val_accuracy'], label='validation accuracy')
plot.legend()
plot.show()
plot.savefig('AccVal_acc')
final_model.save('zature_model.h5') #Modeli saklıyoruz.
