"""
Tipologia più applicata ed usata.
Predirre N valori non solo uno
"""

--- TUTTO IL FILE NON è STATO SALVATO QUINDI GUARDARE LIBRO PER RECUPERARE CODICE!!!


"""
Cosa abbiamo imparato?
- Se abbiamo N possibili valori categorici allora il nostro modello dovrà finire con un layer di N neuroni
- Il nostro network dovrebbe finire con una SOFTMAX activation la quale dirà per ogni valore categorico la probabilità (la somma dell'array finale sarà 1)
- Categorical Crossentropy è quasi sempre utilizzato come loss function la quale tende a minimizzare la distanza di probabilità dal target e il predetto
- Abbiamo due metodologie per gestire le etichette e/o valori target: One-hot encode dove produciamo un array con N valori dove N è il numero di possibilità con 1 e 0 come valori
oppure ogni label N avrà un valore intero diverso. Qualora la scelta ricada su una di queste due anche la metrica sarà diversa.
- Se hai bisogno di classificare un numero elevato di categorie la tua rete NON DOVRA' mai avere un layer con un numero di Nueroni inferiore al numero di casi possibili
altrimenti si avrà perdita di informazione. (eg hai 46 possibili valori categorici, la rete neurale ogni layer dovrà avere un numero di neuroni >= 46)
"""
