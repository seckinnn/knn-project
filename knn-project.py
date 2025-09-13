import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

 # Sonuçlar çizgi grafiği türünde gösterilir
def evaluate_knn(data, target, dataset_name):
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    
    # Hiperparametre aralığını belirle
    param_grid = {'n_neighbors': np.arange(1, 31), 'metric': ['euclidean', 'manhattan', 'minkowski']}
    
    # KNN modeli oluştur
    knn = KNeighborsClassifier()
    
    # GridSearchCV ile en iyi hiperparametreyi bul
    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', verbose=3)
    grid_search.fit(X_train, y_train)
    
    # En iyi hiperparametre ve model
    best_params = grid_search.best_params_
    best_knn = grid_search.best_estimator_
    
    print(f"{dataset_name} veri seti için en iyi parametreler: {best_params}")
    
    # Test seti üzerinde tahmin yap
    y_pred = best_knn.predict(X_test)
    
    # Sonuçları değerlendirme
    print(f"\n{dataset_name} Veri Seti - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n{dataset_name} Veri Seti - Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\n{dataset_name} Veri Seti - Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    
    # Hiperparametre optimizasyon sonuçlarını görselleştirme
    results = grid_search.cv_results_
    plt.figure(figsize=(12, 6))
    for metric in param_grid['metric']:
        mean_test_scores = [results['mean_test_score'][i] for i in range(len(results['params'])) if results['params'][i]['metric'] == metric]
        plt.plot(param_grid['n_neighbors'], mean_test_scores, marker='o', label=f'Metric: {metric}')
    
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} Veri Seti - KNN Hyperparameter Tuning using GridSearchCV')
    plt.legend()
    plt.grid(True)
    plt.show()

# Iris veri setini yükle ve değerlendir
iris = load_iris()
evaluate_knn(iris.data, iris.target, "Iris")

# Digits veri setini yükle ve değerlendir
digits = load_digits()
evaluate_knn(digits.data, digits.target, "Digits")

# Kanser veri setini yükle ve değerlendir
cancer = load_breast_cancer()
evaluate_knn(cancer.data, cancer.target, "Breast Cancer")
