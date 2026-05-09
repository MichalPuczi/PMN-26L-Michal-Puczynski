import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol',
                'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(data_url, names=column_names)
data.replace('?', np.nan, inplace=True)
print(data.head())

print("\n--- DATASET INFORMATION ---")
print(data.info())

#Liczenie braków
print("\n--- MISSING DATA ---")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Cel (0-zdrowy, 1-chory)
data['target'] = (data['target'] > 0).astype(int)

# Podział X i y
X = data.drop('target', axis=1)
y = data['target']

# Podział trening/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cechy liczbowe i kategoryczne
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Potoki przetwarzania
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Braki -> mediana
    ('scaler', StandardScaler())                   # Skalowanie
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Braki -> moda
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first')) # Kodowanie 0-1
])

# Łączenie transformatorów
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

print("Data successfully split and preprocessed!")

# Główny Pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Siatka parametrów
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga']
}

# GridSearch (CV=5, optymalizacja ROC AUC)
print("\nStarting grid search experiments...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Trenowanie
grid_search.fit(X_train, y_train)

# Najlepszy model
best_model = grid_search.best_estimator_

print("\n--- EXPERIMENT RESULTS ---")
print("Best parameters found:")
print(grid_search.best_params_)

print("\n--- TESTING ON NEW DATA ---")
#Predykcja
y_pred = best_model.predict(X_test)

# Prawdopodobieństwo (do ROC)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Raport klasyfikacji
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Macierz błędu
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Krzywa ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Classifier Quality')
plt.legend(loc="lower right")
plt.show()