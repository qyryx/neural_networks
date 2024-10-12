import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sn
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# vytvorime si reprezentacie jednotlivych cifier zakodovane pomocou 0 a 1 v rastri 4x7
# na ich konce pridame zelane vystupy reprezentovane polom, kde na indexe cisla je 1

one = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
two = [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
three = [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
four = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
five = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
six = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
seven = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
eight = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
nine = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
zero = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# cifry spojime do zakladnej jednotky datasetu, tu potom 20x skopirujeme a vytvorime si dataset 200x38

default_set = np.array([one, two, three, four, five, six, seven, eight, nine, zero])
data = np.tile(default_set, (20, 1))

# dataset ulozime do suboru csv

np.savetxt('data.csv', data, delimiter=',', fmt="%i")


# definujeme si funkcie, ktore nam budu vytvarat porusenie a zasumenie testovacich dat

# funkcia swap_values nam podla zadefinovanych % hodnot vymeni hodnoty jednotlivych pixelov v rastri
# 2. argument urcuje pocetnost riadkov a 3. argument urcuje pocet stlpcov ktore maju byt vymenene
# swap_values(data, 20, 10) teda znamena vymen 10% hodnot v riadku v 20% vsetkych riadkov


def swap_values(input_array, row_mutate_pct, value_mutate_pct):
    mutated_array = input_array.copy()

    num_rows = input_array.shape[0]
    num_values_per_row = input_array.shape[1]

    num_rows_to_mutate = int(num_rows * (row_mutate_pct / 100))
    rows_to_mutate = random.sample(range(num_rows), num_rows_to_mutate)

    for row_index in rows_to_mutate:
        row = mutated_array[row_index]

        num_values_to_mutate = int(num_values_per_row * (value_mutate_pct / 100))

        positions_to_mutate = random.sample(range(num_values_per_row), num_values_to_mutate)

        for position in positions_to_mutate:
            row[position] = 1 - row[position]

    return mutated_array

# funkcia add_or_remove_values nam podla zadefinovanych % hodnot prida alebo odoberie 0 a 1
# 2. argument urcuje pocetnost riadkov a 3. argument urcuje pocet stlpcov ktore maju byt ovplyvnene
# add_or_remove_values(data, 20, 10) teda znamena zmen 10% hodnot v riadku v 20% vsetkych riadkov


def add_or_remove_values(input_array, row_mutate_pct, value_mutate_pct):
    mutated_array = input_array.copy()
    num_rows = input_array.shape[0]
    num_values_per_row = input_array.shape[1]
    num_rows_to_mutate = int(num_rows * (row_mutate_pct / 100))

    rows_to_mutate = random.sample(range(num_rows), num_rows_to_mutate)

    for row_index in rows_to_mutate:
        row = mutated_array[row_index]

        num_values_to_mutate = int(num_values_per_row * (value_mutate_pct / 100))
        positions_to_mutate = random.sample(range(num_values_per_row), num_values_to_mutate)

        for position in positions_to_mutate:
            original_value = row[position]

            if original_value == 0:
                row[position] = 1
            else:
                row[position] = 0

    return mutated_array

# funkcia noise_values nam podla zadefinovanych % hodnot zasumi hodnoty, teda vysledne hodnoty sa nachadzaju
# v intervali <0:1:0.1>
# 2. argument urcuje pocetnost riadkov a 3. argument urcuje pocet stlpcov ktore maju byt ovplyvnene
# noise_values(data, 20, 10) teda znamena zasum 10% hodnot v riadku v 20% vsetkych riadkov


def noise_values(input_array, row_mutate_pct, value_mutate_pct):
    mutated_array = input_array.copy()

    num_rows = input_array.shape[0]
    num_values_per_row = input_array.shape[1]

    num_rows_to_mutate = int(num_rows * (row_mutate_pct / 100))
    rows_to_mutate = random.sample(range(num_rows), num_rows_to_mutate)

    for row_index in rows_to_mutate:
        row = mutated_array[row_index]
        num_values_to_mutate = int(num_values_per_row * (value_mutate_pct / 100))
        positions_to_mutate = random.sample(range(num_values_per_row), num_values_to_mutate)

        for position in positions_to_mutate:
            original_value = row[position]

            if original_value == 0:
                new_value = random.uniform(0, 0.5)
            else:
                new_value = random.uniform(0.5, 1)

            row[position] = new_value

    return mutated_array


# nacitame si data zo suboru a rozdelime ich na zavisle a nezavisle

dataset = pd.read_csv('data.csv', header=None)
X = dataset.iloc[:, :28].values
Y = dataset.iloc[:, 28:38].values

# data rozdelime na trenovacie a testovacie v pomere 70/30

X_train = X[:140]
Y_train = Y[:140]
X_test = X[140:200]
Y_test = Y[140:200]

# upravujeme testovacie data vytvorenymi funkciami
# data mozeme vymienat, pridavat/odoberat hodnoty, zasumovat alebo akekolvek kombinacie spomenutych

X_test = swap_values(X_test, 20, 20)
X_test = add_or_remove_values(X_test, 20, 20)
X_test = noise_values(X_test, 30, 50)

# inicializujeme neuronovu siet, nastavime skryte vrstvy, aktivacne funkcie a ich pocet neuronov
# pridame typy optimalizacie a strat a metrik ktore chceme sledovat, nastavime velkost davky a pocet epoch

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=28, activation='relu', input_dim=28))

ann.add(tf.keras.layers.Dense(units=18, activation='relu'))

ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

ann.add(tf.keras.layers.Dense(units=10, activation='softmax'))

ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann1 = ann.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=10, epochs=40)

# vykreslime grafy ktore nam hovoria o uspesnosti trenovania a testovania

plt.plot(ann1.history['accuracy'])
plt.plot(ann1.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# vykreslime grafy ktore nam hovoria o chybe pri trenovani a testovani

plt.plot(ann1.history['loss'])
plt.plot(ann1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# vykreslime maticu v ktorej vidime ako neuronova siet zatriedovala data

Y_predicted = ann.predict(X_test)
cm = confusion_matrix(np.asarray(Y_test).argmax(axis=1), np.asarray(Y_predicted).argmax(axis=1))

labels = unique_labels(Y_test)
df_cm = pd.DataFrame(cm, index=[i for i in labels], columns=[i for i in labels])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
