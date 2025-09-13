import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Sonuçlar renkler türünden gösterilir
# Accuracy sonuçlarını saklamak için global sözlük
accuracy_results = {}

def evaluate_knn(data, target, dataset_name):
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    
    # Özellikleri ölçekle
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Hiperparametre aralığını belirle
    param_grid = {
        'n_neighbors': np.arange(1, 31),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }
    
    # KNN modeli oluştur
    knn = KNeighborsClassifier()
    
    # GridSearchCV ile en iyi hiperparametreyi bul
    grid_search = GridSearchCV(
        knn, param_grid, cv=10, scoring='accuracy', verbose=0, n_jobs=-1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    # En iyi hiperparametre ve model
    best_params = grid_search.best_params_
    best_knn = grid_search.best_estimator_
    
    print(f"\n📊 {dataset_name} veri seti için en iyi parametreler: {best_params}")
    print(f"⏱ Eğitim süresi: {end_time - start_time:.2f} saniye")
    
    # Test seti üzerinde tahmin yap
    y_pred = best_knn.predict(X_test)
    
    # Sonuçları değerlendirme
    print(f"\n{dataset_name} Veri Seti - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n{dataset_name} Veri Seti - Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\n{dataset_name} Veri Seti - Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    
    # Accuracy skorunu global sözlüğe kaydet
    accuracy_results[dataset_name] = accuracy_score(y_test, y_pred)
    
    # Hiperparametre optimizasyon sonuçlarını görselleştirme (heatmap)
    results = pd.DataFrame(grid_search.cv_results_)
    pivot_table = results.pivot_table(
        values='mean_test_score',
        index='param_n_neighbors',
        columns=['param_metric', 'param_weights']
    )
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_table, annot=False, cmap="viridis")
    plt.title(f"{dataset_name} - GridSearchCV Accuracy Scores (CV Ortalama)")
    plt.ylabel("n_neighbors (k)")
    plt.xlabel("Metric & Weights")
    plt.show()


# Iris veri setini yükler ve değerlendir
iris = load_iris()
evaluate_knn(iris.data, iris.target, "Iris")

# Digits veri setini yükler ve değerlendir
digits = load_digits()
evaluate_knn(digits.data, digits.target, "Digits")

# Breast Cancer veri setini yükler ve değerlendir
cancer = load_breast_cancer()
evaluate_knn(cancer.data, cancer.target, "Breast Cancer")


# Tüm datasetlerin accuracy skorlarını özetler
print("\n📌 Accuracy Sonuçları (Test Seti):")
for dataset, acc in accuracy_results.items():
    print(f"- {dataset}: {acc:.4f}")
