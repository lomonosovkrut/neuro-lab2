# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

# Попробуем обучить один нейрон на задачу классификации двух классов

import pandas as pd  # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt  # matplotlib для построения графиков
import numpy as np  # numpy для работы с векторами и матрицами

# Считываем данные 
df = pd.read_csv('data.csv')

# смотрим что в них
print(df.head())

# три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем три признака
X = df.iloc[:, [0, 1, 2]].values

# Признаки в X, ответы в y - посмотрим на плоскости как выглядит задача
plt.figure()
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', marker='x', label='Iris-versicolor')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()

# Переходим к созданию нейрона
def neuron(w, x):
    if (w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]) >= 0:
        predict = 1
    else:
        predict = -1
    return predict

# Проверим как это работает (веса зададим пока произвольно)
w = np.array([0, 0.1, 0.4, 0.2])
print(neuron(w, X[1]))  # вывод ответа нейрона для примера с номером 1

# Теперь создадим процедуру обучения
w = np.random.random(4)  # зададим начальные значения весов
eta = 0.01  # скорость обучения
w_iter = []  # пустой список, в него будем добавлять веса, чтобы потом построить график

for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    w[1:] += (eta * (target - predict)) * xi  # корректировка весов
    w[0] += eta * (target - predict)
    if j % 10 == 0:
        w_iter.append(w.tolist())

# Посчитаем ошибки
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi)
    sum_err += (target - predict) / 2

print("Всего ошибок: ", sum_err)

# Визуализация процесса обучения
xl = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)  # диапазон координаты x для построения линии
yl = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)  # диапазон координаты y для построения линии
xl, yl = np.meshgrid(xl, yl)

# Вычисляем значения для третьего признака
w = np.array(w_iter[-1])  # используем последние веса
x3_grid = -(w[1] * xl + w[2] * yl + w[0]) / w[3]  # уравнение плоскости

# Построим сначала данные на плоскости
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', marker='o', label='Iris-setosa')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', marker='x', label='Iris-versicolor')

# Рисуем разделяющую плоскость
ax.plot_surface(xl, yl, x3_grid, alpha=0.5, color='gray')

ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
ax.legend()
plt.show()