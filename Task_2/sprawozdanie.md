Sprawozdanie nr2 grupa B 

Michał Puczyński 
indeks: 119091
ps1  

1. Regresja Logistyczna

Jest to klasyfikator, który oblicza sumę ważoną cech pacjenta, a następnie przekształca wynik za pomocą **funkcji sigmoidalnej**. Funkcja ta mapuje dowolne liczby na przedział 0 do 1, co interpretujemy jako prawdopodobieństwo wystąpienia choroby. Jest to standard w medycynie dla problemów binarnej klasyfikacji (zdrowy/chory). Cechuje się wysoką **interpretowalnością** – pozwala łatwo zrozumieć, które czynniki (np. cholesterol, wiek) mają największy wpływ na ryzyko zachorowania.
Opis parametrów:
**`C`**: Odwrotność siły regularyzacji. Mniejsze wartości (np. 0.1) silniej "hamują" model, zapobiegając dopasowaniu się do szumu w danych (overfitting).
**`penalty`**: Rodzaj kary nakładanej na zbyt duże wagi cech (`l1` lub `l2`). Pomaga to uprościć model i wyłonić najważniejsze zmienne.
**`solver`**: Algorytm optymalizacyjny (np. `liblinear`, `saga`), który matematycznie szuka najlepszego zestawu wag dla naszych danych.

3. Analiza zbioru danych

Zbiór danych zawiera **303 próbki**. Po przekształceniu problemu na klasyfikację binarną, rozkład klas jest zbliżony do zrównoważonego, co zapewnia stabilne warunki do nauki modelu.
Analiza wykazała braki danych w dwóch kolumnach: 
**`ca`** (liczba zabarwionych naczyń): 4 braki.
**`thal`** (rodzaj defektu serca): 2 braki.

Aby nie tracić cennych danych medycznych, zamiast usuwać wiersze, zastosowano technikę imputacji:
Dla zmiennych liczbowych (jak `ca`) braki uzupełniono **medianą**.
Dla zmiennych kategorycznych (jak `thal`) braki uzupełniono **modą** (najczęściej występującą wartością).

4. Wykonane eksperymenty

W programie wykorzystano narzędzie **GridSearchCV**, które przeprowadziło serię eksperymentów, testując 20 różnych kombinacji parametrów modelu. Każda kombinacja była sprawdzana za pomocą 5-krotnej walidacji krzyżowej.

* Testowano różne siły regularyzacji (**C**): od bardzo silnej (0.01) do słabej (100).
* Sprawdzano dwa rodzaje kar (**penalty**): `l1` oraz `l2`.
* Porównywano efektywność solverów: `liblinear` oraz `saga`.
Najlepszy model został wybrany na podstawie metryki **ROC AUC**, co pozwoliło uzyskać klasyfikator najlepiej oddzielający osoby chore od zdrowych.

5. Wykresy i wizualizacje

W programie zaimplementowano dwie kluczowe wizualizacje oceniające jakość medyczną modelu:

* **Macierz Błędu (Confusion Matrix):** Pozwala na precyzyjne odczytanie liczby poprawnych diagnoz oraz pomyłek. Skupiamy się tutaj na minimalizacji pomyłek typu *False Negative* (chory uznany za zdrowego).
* **Krzywa ROC (Receiver Operating Characteristic):** Ilustruje jakość klasyfikatora na różnych progach decyzyjnych. 
* Kluczowym wskaźnikiem jest **AUC (Area Under Curve)**. 
* Wynik AUC w Twoim programie (ok. 0.90) oznacza, że model ma 90% szans na poprawne rozróżnienie pacjenta chorego od zdrowego, co jest wynikiem bardzo wysokim w diagnostyce medycznej.

6. Wnioski

*Opis techniczny procesu:
**Wczytanie i czyszczenie:** Dane załadowano z UCI, zamieniając znaki `?` na `NaN`.
**Binaryzacja:** Zmienną celu przekształcono na format 0-1 (zdrowy/chory).
**Preprocessing (Pipeline):** * Dla liczb: Uzupełnienie braków medianą i standaryzacja (`StandardScaler`).
    * Dla kategorii: Uzupełnienie braków najczęstszą wartością i zamiana na zera/jedynki (`OneHotEncoder`).
**Modelowanie:** Uruchomiono `GridSearchCV`, aby automatycznie dobrać najlepsze parametry Regresji Logistycznej.

Podsumowanie wyników:
**Wysoka Czułość (Recall):** Model osiągnął bardzo wysoki wskaźnik recall dla osób chorych (ok. 89%). W medycynie oznacza to, że model niezwykle rzadko przeocza faktyczne przypadki choroby (bardzo mało błędów *False Negative*).
**Dokładność (Accuracy):** Wynik na poziomie 85% świadczy o wysokiej wiarygodności ogólnej modelu.
**Stabilność (AUC):** Pole pod krzywą ROC (AUC = 0.90) potwierdza, że klasyfikator bardzo skutecznie odróżnia pacjentów wymagających leczenia od osób zdrowych, co czyni go użytecznym narzędziem wsparcia decyzji lekarskich.