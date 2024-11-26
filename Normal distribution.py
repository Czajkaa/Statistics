from numpy.random import randn
from numpy.random import poisson
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

alpha = 0.05
data1 = 10 * randn(100)
data2 = poisson(5, 100)
alpha = 0.05

W1, p1 = shapiro(data1)
W2, p2 = shapiro(data2)
print("- Test Shapiro-Wilka: data1: W = %.2f, p = %.2f | data2: W = %.2f, p = %.2f" % (W1, p1, W2, p2))
print("Dla testu 1:", end=' ')
if p1 > alpha: print("Nie ma podstaw do odrzucenia hipotezy zerowej - dane pochodzą z rozkładu normalnego.")
else: print("Odrzucamy hipotezę zerową - dane nie pochodzą z rozkładu normalnego.")
print("Dla testu 2:", end=' ')
if p2 > alpha: print("Nie ma podstaw do odrzucenia hipotezy zerowej - dane pochodzą z rozkładu normalnego.")
else: print("Odrzucamy hipotezę zerową - dane nie pochodzą z rozkładu normalnego.")

K1, p1 = normaltest(data1)
K2, p2 = normaltest(data2)
print("\n- Test K^2 D'Agostino: data1: K = %.2f, p = %.2f | data2: K = %.2f, p = %.2f" % (K1, p1, K2, p2))
print("Dla testu 1:", end=' ')
if p1 > alpha: print("Nie ma podstaw do odrzucenia hipotezy zerowej - dane pochodzą z rozkładu normalnego.")
else: print("Odrzucamy hipotezę zerową - dane nie pochodzą z rozkładu normalnego.")
print("Dla testu 2:", end=' ')
if p2 > alpha: print("Nie ma podstaw do odrzucenia hipotezy zerowej - dane pochodzą z rozkładu normalnego.")
else: print("Odrzucamy hipotezę zerową - dane nie pochodzą z rozkładu normalnego.")

A1 = anderson(data1)
A2 = anderson(data2)
print("\n- Anderson-Darling Test: data1: A = %.2f, p = %.2f | data2: A = %.2f, p = %.2f" % (A1.statistic, A1.critical_values[2], A2.statistic, A2.critical_values[2]))
print("Dla testu 1:", end=' ')
if A1.statistic > A1.critical_values[2]: print("Odrzucamy hipotezę zerową - dane nie pochodzą z rozkładu normalnego.")
else: print("Nie ma podstaw do odrzucenia hipotezy zerowej - dane pochodzą z rozkładu normalnego.")
print("Dla testu 2:", end=' ')
if A2.statistic > A2.critical_values[2]: print("Odrzucamy hipotezę zerową - dane nie pochodzą z rozkładu normalnego.")
else: print("Nie ma podstaw do odrzucenia hipotezy zerowej - dane pochodzą z rozkładu normalnego.")
print()