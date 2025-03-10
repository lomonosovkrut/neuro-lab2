# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:27:38 2025

@author: ПользовательHP
"""

# Рассмотрим основы языка python
# Программа на языке python обычно состоит из нескольких блоков

# 1 блок импорта пакетов (библиотек, модулей) обычно находится в начале файла,
# но импорты работают в любом месте кода

# импорт может быть всей библиотеки
import time

# или отдельных функций (модулей, объектов и т.п.) из библиотеки
from random import randint

# часто библиотека импортируется под псевдонимом для более краткого к ней обращения
import math as M

# если при импорте что-то пошло не по плану, в консоли вы увидите ошибку
# import liba
# ModuleNotFoundError: No module named 'liba' - означает что библиотека не установлена
# ImportError: cannot import name 'randint1' from 'random' - означает что указанной функции нет в библиотеке

# 2 После блока импортов размещается код функций

# для создания функции указывается ключевое слово def, после которого идет 
# название функции и в скобках параметры двоеточие завершает объявление функции

# запускать блоком! #
def sumnum(a, b):
    c = a + b     # после двоеточия следует код функции
    return c
#####################

## Обратите внимание, что вложенность кода в python задается пробелами или табом##

# 3 После кода функций следует блок основного кода программы
# Выполнение задания №1

# Исправление: Инициализируем список с помощью метода append()
spisok = []
sum = 0
for i in range(100):
    spisok.append(randint(0, 20))  # Добавляем элементы в список
    if spisok[i] % 2 == 0:         # Проверяем, является ли элемент четным
        sum += spisok[i]
        print(sum)

# к элементам (функциям) импортированной библиотеки обращение происходит через точку
print(time.time())

print(M.sqrt(16))

# к отдельным функциям напрямую
print(randint(0, 10))

# значения переменным присваиваются через знак равенства
a = 5

b = randint(0, 10)

t = time.time()

# допускается множественное присвоение переменных
c = d = e = 10

# существуют различные типы данных 
# но python поддерживает динамическую типизацию
# это означает, что переменная может менять тип в процессе выполнения программы
# понаблюдайте за изменением типа переменной a

# целое число
a = int(5)
print(a)
print(type(a))

# дробное число
a = float(.5)
print(a)
print(type(a))

# строка
a = '.5'
print(a)
print(type(a))

# приведение типов осуществляется прямым указанием типа перед значением
a = str(.5)
print(a)
print(type(a))

# строки с обеих сторон ограничиваются одинарными или двойными кавычками
a = ".5"
print(a)

# конкатенация строк делается через сложение
b = " - это дробное число"
print(a + b)

# длина строки (и не только строки) вычисляется оператором len
c = a + b
print(len(c))

# к элементам строки можно обратиться по индексу
print(c[10])

# можно делать срезы — получение какой‑то части строки, которая ограничена 
# индексами
print(c[10:15])

print(c[:15])

print(c[15:])

# в срезах можно задавать шаг
print(c[0:15:3])

# массивов по умолчанию нет, но есть списки
# к элементам списка также можно обращаться по индексу и делать срезы
a = [7, 5, 0, 2, 3]

print(a[0])

print(a[3:])

# списки могут хранить любые значения
b = ['Пенза', 'Самара', 'Саратов', 12, 33]
print(b)

# в список можно добавить элемент
b.append(0.589)
print(b)

# или соединить с другим списком
b.extend([1, 2, 3])
print(b)

# список можно отсортировать 
a.sort()
print(a)

# циклы бывают нескольких видов

# для выполнения цикла заданное количество итераций
num = 0
for i in range(5):
    num = num + 1       # вложенность задается табуляцией (4 пробела)
    print(num)

# в цикле могут перебираться значения из списка
lst = [1, 4, 9, 11, 12]
for i in lst:
    print(i % 2)

# цикл while
k = 10
while k > 5:
    print(k)
    k = k - 1

# оператор in может использоваться для проверки наличия значения в списке
if 5 in a:
    print('Есть')
else:
    print('Нет')