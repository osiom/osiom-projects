"""
Tipologia più applicata ed usata.

Questo caso è diverso dal precedente. Qui dovremo predirre un valore continuo e non più uno categorico.

Valore dei prezzi delle case sulla base di features (eg Tasso criminalità zona). Alcuni valori di features avranno intervalli 0-1 altri 1-100 o ancora 1-12
"""

from keras.datasets import boston_housing

(train_data, train_targets), (test_data,test_targets) = boston_housing.load_data()
train_data.shape #404 possibili casi con 13 features numeriche
test_data.shape #102 possibili casi con 13 features numeriche

"""
Le features numeriche sono il tasso criminalità, il numero medio di stanze, l'accessibilità alla superstrada e così via
"""

train_targets #questi sono i prezzi delle varie case del train data set

"""
Risulta problematico passare alla rete neurale valori che possono ricevere diversi range (scale da 1-10 o da 1-100)
La rete può lo stesso essere allenata ma il training risulterebbe piu difficoltoso data l'eterogeneità.

La best-practice è che tali dati vengano normalizzati. Per ogni feature in input si toglie la media della feature e si divide per la deviazione standard.

Questo viene fatto in Numpy
"""

mean = train_data.mean(axis=0) #creo l'array delle medie
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean #normalizzazione del test viene fatto con media e std del train dataset MAI fare operazioni con set di test
test_data /= std

"""
Visto che il dataset utilizzato è veramente piccolo utilizzeremo solo 2 layer ognuno con 64 nodi.
In generale meno dati hai più le probabilità di overfitting sono alte.
Per evitare ciò meglio usare una rete neurale piccola
"""

from keras import models
from keras import layers

def build_model():

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) #il modello finisce con un solo valore senza activation function (sarà lineare). Tipico setup per una regression classification
    #Usando una funzione di attivazione sull'ultimo modello potrebbe creare constraint in fase di predizione eg se usassimo una sigmoide sarebbe solo un valore tra 0 ed 1
    #In questo modo il risultato sarà lineare e potrà essere qualsiasi valore
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    #Compile del modello con una mse loss function (mean squared error) ovvero il quadrato della differenza tra la prediction ed il target (most used in regression problems)
    #Monitoriamo tramite la mae ovvero Mean Absolute Error ovvero la differenza tra il predetto ed il target
    return model

"""
Visto che il set di dati in questo caso è troppo piccolo se dovessimo dividere il training in train e validation
friniremmo per avere solo un numero veramente limitato di valori come validation.
Questi inoltre sarebbero influenzati da ciò che viene usato come training e set diversi di training produrrebbero diversi risultati
cosi come diversi validation potrebbero avere risultati diversi.

Una strategia utilizzata è la K-FOLD validation. Cosa fa?
Divide il dataset in base ad un numero K. A rotazione avremo diversi gruppi di analisi per esempio:
K = 3
il nostro data set sarà diviso in 3 sub dataset dove:
- In una rete (la rete avrà sempre la stessa configurazione) il primo set di dati sarà di validation e gli altri due di Training
- In un'altra rete il secondo set di dati sarà di validation ed il primo ed il terzo di Training
.. e cosi via

Questo metodo viene utilizzato spesso con regression problem e quando il dataset risulta piccolo
"""

import numpy as np

k = 4
num_val_samples = len(train_data)/4
num_epochs = 100
all_scores = []

"""
Dividiamo in diversi set (4 per la precisione). Ogni volta avremo un diverso train e validation set.
Passeremo al nostro modello (definito nella funzione) a ruota tutte le possibili combinazioni e salveremo il risultato come score.
"""

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i+1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i+1) * num_val_samples:]],
        axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,epochs=num_epochs,batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

"""
Come si può notare le diverse esecuzioni hanno diversi validity score. La media di questi è un valore molto più affidabile
piuttosto che considerare solamente un singolo.

Il nostro modello ha quindi in media un MAE di 3 (essendo in migliaia siamo distanti di circa 3000 dollari dal prezzo)
3000 Dollari sono tanti considerando un prezzo di una casa dai 10K ai 50K, cerchiamo di migliorare.
"""

import numpy as np

k = 4
num_val_samples = len(train_data)/4
num_epochs = 500
all_mae_histories = []

"""
Dividiamo in diversi set (4 per la precisione). Ogni volta avremo un diverso train e validation set.
Passeremo al nostro modello (definito nella funzione) a ruota tutte le possibili combinazioni e salveremo il risultato come score.
"""

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i+1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i+1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,epochs=num_epochs,batch_size=1, verbose=0)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)

"""
Abbiamo deciso di salvare il MAE ad ogni ripetizione qui e non più al termine di ogni fold processato

Ora possiamo proiettare la mae al termine di ogni epoca e vedere come è il risultato

Ma prima dobbiamo calcolare per ogni epoca la media delle varie fold.
"""

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

import matplotlib.pyplot as plt

plt.plot(range(1,len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epoch')
plt.ylabel('Validation MAE')
plt.show()

"""
Troppo dettagliata dobbiamo rimpicciolire e riformattare il grafico quindi:
- Escludiamo i primi valori che sono su una scala diversa
- Sostituiamo ogni punto con una media mobile esponenziale del punto precedente per ottenere una curva più piatta
"""

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)

    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epoch')
plt.ylabel('Validation MAE')
plt.show()

"""
Ora facciamo la train finale del modello dopo aver fatto vari tentativi. Abbiamo visto dai grafici che dopo 80 epoche il modello overfitta quindi proviamo a diminuirle

Infine facciamo l'evaluation
"""

model = build_model()
model.fit(train_data, train_targets, epochs = 80, batch_size = 16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mse_score
test_mae_score #Siamo sempre off target di 2.5K dollari

""""
Cosa abbiamo imparato?
- La regressione è fatta usando diverse loss function -> MSE è la piu usata comunemente
- Similarmente anche le metriche sono diverse, la più comune è MAE
- Quando abbiamo più features dobbiamo omologare le scale in ordine da rendere il training più semplice
- Quando abbiamo pochi dati disponibili utilizziamo K-Fold validation
- Quando il training set è di pochi dati è meglio fare reti neurali piccole per evitare overfitting
"""""
