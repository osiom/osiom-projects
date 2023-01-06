#%%
import os, shutil

original_dataset_dir = '/Users/matteoosio/Downloads/dogs-vs-cats/train'
base_dir = '/Users/matteoosio/Documents/deeplearning-python/cats_and_dog_small'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
#os.mkdir(test_dir)

#%%
train_cats_dir = os.path.join(train_dir,'cats')
train_dog_dir = os.path.join(train_dir,'dog')
#os.mkdir(train_cats_dir)
#os.mkdir(train_dog_dir)

validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dog_dir = os.path.join(validation_dir,'dog')
#os.mkdir(validation_cats_dir)
#os.mkdir(validation_dog_dir)

test_cats_dir = os.path.join(test_dir,'cats')
test_dog_dir = os.path.join(test_dir,'dog')
#os.mkdir(test_cats_dir)
#os.mkdir(test_dog_dir)

#%%
'''Primi 1000 record come train'''
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for name in fnames:
    src = os.path.join(original_dataset_dir, name)
    dst = os.path.join(train_cats_dir,name)
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for name in fnames:
    src = os.path.join(original_dataset_dir, name)
    dst = os.path.join(train_dog_dir,name)
    shutil.copyfile(src,dst)

'''Successivi 500 record come validation'''
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for name in fnames:
    src = os.path.join(original_dataset_dir, name)
    dst = os.path.join(validation_cats_dir,name)
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for name in fnames:
    src = os.path.join(original_dataset_dir, name)
    dst = os.path.join(validation_dog_dir,name)
    shutil.copyfile(src,dst)

'''Successivi 500 record come test'''
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for name in fnames:
    src = os.path.join(original_dataset_dir, name)
    dst = os.path.join(test_cats_dir, name)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for name in fnames:
    src = os.path.join(original_dataset_dir, name)
    dst = os.path.join(test_dog_dir, name)
    shutil.copyfile(src, dst)

#%%

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dog_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dog_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dog_dir)))

#%%
#Start to create the model il quale avrà 4 Conv net (+4 MaxPooling), un layer per Flatten ed infine concluderemo con una sigmoide a 2 neuroni 0/1

from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

"""
Compilation step:
- Optimizer = RMSprop
- Loss function = Binary crossentropy
- Metrics to Monitor = Accuracy
"""
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#%%
"""
Data Preparation
- Read the pictures
- Decode the JPEG to RGB Grid of pixels
- Convert into floating-point tensors
- Rescale the pixel values to the 0:1 interval for better processing
Keras offre una libreria che fa al caso nostro con la classe ImageDataGenerator

This produce a generator where each element is a batch of images that are gonna be used to feed the Neural Network
with the fit_generator method.
"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255) #rescale le immagini di 1/255
test_datagen = ImageDataGenerator(rescale=1./255) #rescale le immagini di 1/255

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150), #resize di tutte le immagini in 150 x 150
    batch_size=20,
    class_mode='binary' #visto che uso la sigmoide ho bisogno di binary labels 0/1
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

#%%

"""
Il generator continua a leggere le immagini dalla cartella in modo infinito, per questo motivo quando facciamo il fit_generator
dobbiamo specificare anche dopo quanti step la epoca viene cambiata (e quindi i pesi/bias aggiustati).
"""

import sys
from PIL import Image

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

#Good practice to save the model

model.save('cats_and_dogs_small_1.h5')

#%%

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure

#%%
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

#%%
"""
Andiamo velocemente in overfitting dato il numero di campioni ristretti.
Due modi visti in precedenza per avoid l'overfitting sono: L2 Regularization (o weight decay) ed il drop out.
Qualcosa di più specifico per la computer vision è la Data Augmentation.
Questo non fa altro che procedere a trasformare ogni singolo campione nel nostro data set in modo che il nostro modello
non processerà due volte la stessa immagine.

ImageDataGenerator instance permette di configurare il numero di transformazioni random che verranno performate su un'immagine
"""

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#%%
"""
Come visto qui abbiamo solo alcune delle opzioni disponibili dove:
- Rotation range è l'angolatura random di rotazione dell'immagine (da 0 a 180)
- Widht_shift e height_shift sono intervalli random all'interno della quale l'immagine viene traslata
- Shear range applica una shearing transformation randomly ( taglio (?))
- zoom_range è uno zoom in nell'immagine
- horizontal_flip randomly flipping metà dell'immagine orizzontalmente. è particolarmente indicato quando non ci sono assunzioni
per la simmetricità dell'immagine (eg real world pictures)
- fill_mode è la strategia utilizzata per il fill dell'immagine quando vengono aggiunti pixel dopo una rotazione o widht/height shift
"""

from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[5] #prendiamo solo una immagine per dimostrare la data augmentation

img = image.load_img(img_path, target_size=(150,150)) #legge immagine e fa il resize
x = image.img_to_array(img) #Converte l'immagine in numpy array con shape 150,150,3

x = x.reshape((1,)+x.shape) #reshape it to 1,150,150,3

#genera un batch di immagini random transformate. loops infinito visto il generatore ma dopo 4 mi blocco
i=0
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

#%%
"""
Se ora provo a addestrare un nuovo network con la data-augmentation configuration il network non vedrà la stessa foto due volte
anche se la correlazione delle immagini sarà la stessa visto che, avremo un numero piccolo di immagini originale e non produrremo
nuova informazione ma ri-useremo quella già esistente. Non ci libereremo quindi, almeno non del tutto, dell'overfitting

Per combatterlo possiamo aggiungere un layer di Dropout
"""
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) #0.5 dei nodi saranno spenti per aggiungere drop out

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc']
              )
#%%
"""
Addestriamo la nuova configurazione passando il data-augmentation ed aggiungendo il dropout
"""

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255) #Note that Validation data SHOULD NOT be augmented

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150), #reshape the images to 150,150
    batch_size=32,
    class_mode='binary' #because the binary classification
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_2.h5')

#%%

"""
Aggiustando un po i parametri e/o grazie alla data augmentation siamo riusciti ad arrivare a circa l'87% di accuracy
ma purtroppo non si riesce a migliorare visto che dobbiamo ogni volta addestrare una network from scratch.
Per questo motivo ora proviamo a passare ad un pretrained model.

A volte esistono dei modelli già creati che sono stati addestrati su dataset piu ampi. Possiamo quindi riutilizzarli.
Normalmente prendiamo solamente la Convolutional Base part del modello (max pooling + convnet) perchè la parte finale, 
il classificatore, è specifico per il modello addestrato.
Inoltre, piu si va in profondità nella convolutional base part più il modello apprenderà larghi patterns (orecchie, occhi ecc)
anche questi sono parecchio specifici sulla base del modello in sè quindi conviene sempre prendere solo la prima parte
della convolutional base part ovvero quella che riconosce pattern più generalizzati (colori, texture, edges)
"""
#%%
from keras.applications import VGG16 #Modello pre-trained basato su ImageNet, classifica diverse cose tra cui anche cani e gatti
#%%
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))

conv_base.summary()
"""
Qui possiamo vedere 3 parametri passati dove il piu importante è il secondo:
Settando a falso gli diciamo di non includere la parte finale il fully-dense connected layer
"""

"""
Ora che abbiamo tutta la nostra parte di Conv Base possiamo fare due cose:
- Ingestion dei nostri dati e salvataggio dell'output per poi inserirlo in un modello stand-alone con la fully dense connected
per produrre i risultati (piu veloce e cheap in termine di risorse)
- Aggiungere on top of the stack un layer fully dense connected alla nostra conv base per produrre cosi un modello completo.
Questo ci permette di utilizzare Data Augmentation perchè ogni immagine passa dalla rete neurale. Ovviamente è piu costoso
"""

"""
Fast Feature Extraction Without Data Augmentation (Caso 1)

Inserisco i dati nella conv base e salvo il risultato, con la sua label, in un array che verrà inserito poi in un
nuovo modello fully connected
"""
#%%
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1) * batch_size] = features_batch
        labels[i * batch_size : (i+1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# Salvo le features ingested nel conv base
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 2000)
test_features, test_labels = extract_features(test_dir, 2000)

train_features = np.reshape(train_features, (2000,4*4*512))
validation_features = np.reshape(validation_features, (2000,4*4*512))
test_features = np.reshape(test_features, (2000,4*4*512))

#%%
# Ora creo il modello fully connected finale
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu',input_dim = 4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features,validation_labels))

#%%

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend

plt.show()
#%%
"""
Now we are gonna see the second case -> Data Augmentation while stacking on top of the conv base the fully dense layer
that is gonna be used to predict our model.

This is much slower and and more expensive but allow us to use the data augmentation while training -> feeding more data
and leading to more accurate ones

This second step non sarà ripreso qui su PyCharm perchè si esegue solo se si ha accesso ad una GPU
altrimenti si segue lo step 1.
"""
#%%
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
#%%

"""
Prima di procedere al training bisogna congelare i pesi del modello conv_base poichè col nuovo training potrebbero essere
modificati e quindi perderemmo tutto il vantaggio del modello già addestrato.

In questo modo solo i pesi dei neuroni added on stack potranno essere aggiustati
"""
#%%
print('number of parameters that cant be trained', len(model.trainable_weights))
conv_base.trainable = False
print('number of parameters that cant be trained', len(model.non_trainable_weights))
#%%

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator= train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size = 20,
    class_mode='binary'
)
validation_generator= test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size = 20,
    class_mode='binary'
)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc']
              )
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
#%%

"""
Altra tecnica utilizzata viene chiamata fine tuning il chè significa scongelare ed addesrtare alcuni layer già addestrati dal modello
questo di solito avviene con i layer nella parte superiore (non quelli iniziali visto che l'errore di propagazione poi sarebbe proporzionalmente grande).

Questo avviene:
- Aggiungo i miei nuovi layer
- Freeze della base del network
- Addestro la parte aggiunta da me
- Sblocco/Unfreeze alcuni layers finali
- Addestro di nuovo il tutto combinato

Perchè unfreeze i layer piu alti?
- L'errore propagato se dovessi cambiare i piu bassi sarebbe troppo grande - inoltre più parametri addestro più rischio di cadere in overfitting
- Layer piu alti hanno pattern piu complessi della quale possiamo "disfarci"

Di solito si fine-tuning degli ultimi max 2/3 layersss1
"""

"""
Ora inizieremo a vedere come la ConvNet funziona.
La maggior parte delle volte i modelli utilizzati sono delle black box ma non per la convnet, qui possiamo:
- Identificare come sono visualizzati gli output dei layers
- Visualizzare i pattern riconosciuti (filters)
- Visualizzare una heatmaps delle parti d'interesse di una immagine, che permettono di identificare la classe
"""

"""
1 - Visualizzare le attivazioni intermedie
"""
#%%
from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')
model.summary()
#%%
"""
Carico l'immagine e la processo
"""
img_path = '/Users/matteoosio/Documents/deeplearning-python/cats_and_dog_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image #pre processa immagini in 4D tensor
import numpy as np

img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)
#%%
"""
Stampo l'immagine
"""
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()
#%%
"""
For one single image we have multiple activations at each single layer.
In a 32 × 32 image , dragging the 5 × 5 receptive field across the input image data with
a stride width of 1 will result in a feature map of 28 × 28 output values or 784 distinct activations per image.

Basically this feature map shows how many times a neuron is fired off-or how many different receptive fields will be formed
"""

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(input=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)
#%%
"""
Stampo il risultato

del quarto canale per l'attivazione del primo layer del modello originale
"""

import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0, :, :, 22], cmap='viridis')
plt.show()
#%%
"""
Now let's try to print all the output processed by the first layer
for the same image.
"""
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features//images_per_row
    display_grid = np.zeros((size*n_cols,images_per_row*size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:,:,col*images_per_row+row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size:(col+1)*size,
                        row*size:(row+1)*size] = channel_image
    scale = 1./size
    plt.figure(figsize=(scale * display_grid.shape[1],
               scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto',cmap='viridis')
    plt.show()
#%%
"""
Come si può vedere dal risultato prodotto:
- I primi layer mantengono quasi completa l'informazione dell'immagine
- Piu layer ci addentriamo piu l'immagine presenta meno informazioni riguardo
il contesto dell'immagine ma più informazioni riguardanti elementi dell'immagine
- meno nodi si attivano piu in dettaglio si va, questo perchè i "filtri"
non si attivano non trovando corrispondenza

Cosa abbiamo imparato? L'informazione estratta dai layer diventa sempre piu astratta piu in profondità si va


La stessa cosa succede all'uomo, quando osservi qualcosa e provi a ridisegnarla
(tipo una bicicletta che tu hai probabilmente visto migliaia di volte)
Il tuo cervello memorizza solo un'astrazione di essa e non sarà mai preciso il disegno
"""

"""
2 - Visualizzare i filtri della ConvNet

Visualizzare i pattern imparati i quali ogni filtro applicato si intende di rispondere

Si attualizza con il gradient ascent in input space.

Si costruisce la loss function che massimizza il valore di un dato filtro nella conv net
e poi si utilizza lo stochastic gradient descent per aggiustare i valori dell'input
per massimizzare l'activation value

Per ogni input layer e filter ritorniamo una valida immagine tensore
che rappresenta il pattern che massimizza l'attivazione del filtro specifico
per ottenere il filtro specifico stesso.

Possiamo quindi visualizzare ogni filtro in ogni layer.
Questo ci insegna tanto riguardo ai filtri perchè:
- ogni filtro è la combinazione dei filtri precedenti
- Ogni filtro in ogni livello ha diversi pattern

Più il network sarà profondo e più i filtri imparati saranno complessi:
- i filtri nei primi layer sono semplici archi e colori direzionali
- i filtri nei layer medi comprendono semplici texture formate dagli archi delle precedenti
- i filtri nei layer piu elevati iniziano ad imparare più complesse texture e immagini naturali (occhi, orecchie etc)
"""

"""
3 - Visualizzare la heatmap delle attivazioni per la classe

Un ultima modalità di debugging e specifica per le mis-classificazioni è la 
produzione di una heatmap per le attivazioni della classe:
Per esempio, trovare tutti gli output che si attivano quando la classe Gatto è predetta.

Questo avviene tramite il Grad-CAM ovvero l'identificazione delle parti nell'immagine
che sono the most cat-like.

Permettendo cosi di spiegare:
- Per quale motivo il modello ha predetto un gatto
- Dove il gatto è posizionato nell'immagine
"""