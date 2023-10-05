import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Загрузка набора данных о винах
wine = load_wine()
X = wine.data
y = wine.target
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Выбор значения k (количество ближайших соседей)
k = 5

# Создание и обучение модели k-NN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
# Предсказание классов для тестовых данных
y_pred = knn.predict(X_test)
# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Вывод отчета о классификации
print("Classification Report:\n", classification_report(y_test, y_pred))