import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

iris = load_iris()
dane = pd.DataFrame(iris.data, columns=iris.feature_names)
praw_gatunki = iris.target

model = KMeans(n_clusters=3, random_state=42, n_init=10)
wyniki= model.fit_predict(dane)
dopasowane = np.zeros(len(wyniki))

for i in range(3):
    kwiaty_w_grupie = praw_gatunki[wyniki== i]
    najcz_gatunek = np.bincount(kwiaty_w_grupie).argmax()
    dopasowane[wyniki== i] = najcz_gatunek

#metryki
dokladnosc = accuracy_score(praw_gatunki, dopasowane)
precyzja = precision_score(praw_gatunki, dopasowane, average='macro')
czulosc = recall_score(praw_gatunki, dopasowane, average='macro')
f1 = f1_score(praw_gatunki, dopasowane, average='macro')

print("Wyniki grupowania")
print("Accuracy:", round(dokladnosc, 4))
print("Precision:", round(precyzja, 4))
print("Recall:", round(czulosc, 4))
print("F1-score:", round(f1, 4))

print("Generowanie wykresu:")
tsne = TSNE(n_components=2, random_state=42)
dane_2d = tsne.fit_transform(dane)

plt.scatter(dane_2d[:, 0], dane_2d[:, 1], c=dopasowane, cmap='viridis')
plt.title("Grupowanie irysow - TSNE")
plt.xlabel("Oś X")
plt.ylabel("Oś Y")

plt.savefig('wykres_tsne.png')
plt.show()