"""
Tipologia più applicata ed usata.
Soltanto due valori T o F, Positive or Negative, 1 or 0
"""

from keras.datasets import imdb

(train_data, train_labels), (test_data,test_label) = imdb.load_data(num_words=10000) #the num_words parameter means you will keep the top 10k most occurred words in training data
train_data[0] #list of integer for each integer we have 1 word
train_labels[0] #1 or 0 to specify if is positive or Negative

# Since i'm restricting to index of word 10K i won't have any word with integer > 10k
max([max(sequence) for sequence in train_data])

# Here i can take back the conversion integer to word:
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.iteritems()])

"""
Non si può passare una lista di integer alla rete neurale ma deve essere convertita in tensore
2 Modi per farlo:
- riempi la lista per farle avere tutte la stessa lunghezza quindi avrò una matrice (numero_casi, word_index)
- one-hot encode la tua lista (una review) diventa un vettore di 0 ed 1 quindi dove 1 è quando ho la parola nella recensione e 0 tutto il resto quindi avrà 10k colonne possibili
Procederemo con quest'ultimo
"""

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension)) # create 0 matrix with 25K rows (len of train data) and 10K columns (possible values)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_label).astype('float32')

"""
La scelta degli strati e del numero di neuroni è cio che determina una buona rete neurale e un buon network architect.
Scegliere troppi strati e/o troppi neuroni può comportare a pattern indesiderati e resource expensive operation. Imparare è parte dell'esperienza.

avremo 3 layer
- layer 1 avrà 16 neuroni fully connected
- layer 2 avrà 16 neuroni fully connected
- layer 3 verrà usato per produrre l'output.

I primi due layer utilizzeranno RELU come funzione di attivazione (taglia fuori tutti valori negativi Rectified linear unit)
L'ultimo utilizzerà la sigmoide come funzione di attivazione (produce valore tra 0 ed 1 interpretabile come probabilità)
"""

from keras import models
from keras import layers

model = models.Sequential() #istanzio classe modello
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # Creo layer 1 con 16 neuroni, attivazione relu fully dense e input ovviamente una matrice di 10000 possibili valori
model.add(layers.Dense(16, activation='relu')) # Creo layer 2 con 16 neuroni, attivazione relu fully dense e input ovviamente input shape è output del precedente (definito automaticamente)ù
model.add(layers.Dense(1,activation='sigmoid')) # Creo layer 3 con 1 solo neurone con attivazione sigmoid, esso stamperà il risultato

"""
Una volta creato il modello dobbiamo specificare:
- loss function
- optimizer
Per problemi binari (0,1) come risultato possiamo usare una binary_crossentropy la quale misura la distanza di probabilità, in questo caso quella ipotizzata e/o la vera test_label

Come optimizer useremo rmsprop
"""

model.compile(optimizer='rmsprop', #passo stringa perchè gia conosciuta da keras, potre passare mia funzione
              loss='binary_crossentropy', #passo stringa perchè gia conosciuta da keras, potre passare mia funzione
              metrics=['acc'] #altre metriche stampate in fase di training
            )

"""
per monitorare l'accuracy del modello mai visto fino ad ora creeremo un set di validazione prendendo 10K rows dal training model
"""

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partia_y_train = y_train[10000:]

"""
procederemo al train con 20 epoche (20 volte ripasso gli stessi dati) in mini-batches di 512 elementi (rows) alla volta. Allo stesso modo monitoreremo la loss e accuracy dei 10K primi valori

Il risultato sarà salvato in una variabile History che contiene tutto lo storico di quello che è accaduto durante il training.

"""

history = model.fit(partial_x_train, partia_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()
# 4 entries una per metrica analizzata e monitorata durante il training.
# Ora possiamo usare matplot lib per proiettare la history del nostro risultato

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['acc']) + 1)

import matplotlib.pyplot as plt

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title('Training and Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

"""
Quello che succede qui è particolare:
La loss del training test diminuisce ed è quello che ci aspetteremmo dal gradient boost descending ma, purtroppo la validation aumenta.
Alla quarta iterazione qualcosa inizia a cambiare e l'accuracy della validitation anzichè aumentare resta stabile per poi diminuire leggeremente, cosa succede?
Quello che dicevamo prima, un modello che performa bene anzi ottimale su un test non è per forza un buon modello!
Qui siamo caduti in overfitting ovvero dopo la seconda epoca il modello ha memorizzato test-label per il training set.
In questo caso per prevenire overfitting potrei ridurre il numero di epoche a 3 (prima di quando inizia l'overfitting)
"""

"""
Ripartiamo da capo con meno epoche
"""

model = models.Sequential() #istanzio classe modello
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # Creo layer 1 con 16 neuroni, attivazione relu fully dense e input ovviamente una matrice di 10000 possibili valori
model.add(layers.Dense(16, activation='relu')) # Creo layer 2 con 16 neuroni, attivazione relu fully dense e input ovviamente input shape è output del precedente (definito automaticamente)ù
model.add(layers.Dense(1,activation='sigmoid')) # Creo layer 3 con 1 solo neurone con attivazione sigmoid, esso stamperà il risultato

model.compile(optimizer='rmsprop', #passo stringa perchè gia conosciuta da keras, potre passare mia funzione
              loss='binary_crossentropy', #passo stringa perchè gia conosciuta da keras, potre passare mia funzione
              metrics=['acc'] #altre metriche stampate in fase di training
            )
history = model.fit(x_train, y_train, epochs=4, batch_size=512)
#ora valutiamo il modello sul validation set
results = model.evaluate(x_test, y_test)
#questo approccio un po troppo "alla buona" essendo un esercizio produce un'accuracy dell'88%, potemmo essere in grado di arrivare ad un 95%

"""
una volta creato il modello lo vogliamo usare in modalità predicted, con nuovi dati.

Dove valori vicino ad 1 sono commenti positivi mentre vicino a 0 sono negativi. Quelli tra 0.8 e 0.4 il modello non è sicuro del prodotto
"""
model.predict(x_test)


"""
Ora, per cercare di migliorare l'accuracy si deve giocare con:
- Aumentare il numero di layer
- Aumentare il numero di hidden units
- Cambiare la loss function
- Provare a cambiare le activation function tanh anziche la relu
"""
