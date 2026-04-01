Sprawozdanie nr1 grupa B 

Michał Puczyński 
indeks: 94756 
grupa ps1  

1. Analiza danych
Przeanalizowałem zbiór danych Iris. Zbiór składa się ze 150 próbek kwiatów, które należą do 3 różnych gatunków. Każdy kwiat jest opisany za pomocą 4 cech liczbowych. Zbiór nie posiada żadnych wartości null.

2. Grupowanie
Do znalezienia grup w danych wykorzystałem algorytm K-Means.

Uzyskałem wyniki:
Accuracy: 0.8933
Precision: 0.9072
Recall: 0.8933
F1-score: 0.8918

Wyniki na poziomie prawie 90% pokazują, że K-Means bardzo dobrze poradził sobie z podzieleniem kwiatów, mimo że nie znał na początku ich prawdziwych gatunków.

3. Wizualizacja
Każdy kwiat ma 4 cechy, więc żeby móc narysować wykres 2D, użyłem algorytmu T-SNE do zmniejszenia liczby wymiarów. Wykres znajduje się w pliku ![alt text](wykres_tsne-1.png)
