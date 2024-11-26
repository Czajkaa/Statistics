from numpy.random import randn
from numpy import mean, std, sqrt
from scipy.stats import sem, t
from scipy.stats import ttest_ind, ttest_rel


# Dane wejściowe
data1 = 10 * randn(100) + 20
data2 = 10 * randn(100) + 25
mean1, mean2 = mean(data1), mean(data2)


# t-student dla prób niezależnych
std1, std2 = std(data1), std(data2, ddof=1)
n1, n2 = len(data1), len(data2)
se1, se2 = std1/sqrt(n1), std2/sqrt(n2)
d_se1, d_se2 = se1 - sem(data1), se2 - sem(data2)
sed = sqrt(se1 ** 2.0 + se2 ** 2.0)

t_stat = abs((mean1 - mean2) / sed)
df = n1 + n2 - 2
alpha = 0.05
cv = t.ppf(1.0 - alpha, df)
p = (1 - t.cdf(abs(t_stat), df)) * 2

print("\nTest t-studenta dla prób niezależnych")
print(f"\nŚrednia dla data1 wynosi: {mean1}, a dla data2 wynosi: {mean2}")
print(f"Odchylenie standardowe dla data1 wynosi: {std1}, a dla data2 wynosi: {std2}")
print(f"Błąd standardowy dla data1 wynosi: {se1} ({d_se1}), a dla data2 wynosi: {se2} ({d_se2})")
print("\nWartości z obliczeń")
print(f"Wartość dla testu t-studenta (prawostronny) dla próbek niezależnych wynosi: {t_stat} dla {df} stopni swobody")
print(f"Wartość krytyczna dla danych parametrów wynosi: {cv}, a wartość p: {p}")
if t_stat <= cv:
    print(f"Pozostaje hipoteza zerowa ponieważ {t_stat} <= {cv}")
else: 
    print(f"Przyjmujemy hipotezę alternatywną ponieważ {t_stat} > {cv}")
if p > alpha:
    print("Pozostaje hipoteza zerowa, czyli średnie są równe")
else:
    print("Przyjmujemy hipotezę alternatywną, czyli środki są równe")

t_stat, p = ttest_ind(data1, data2)

print("\nWartości z funkcji SciPy")
print(f"Wartość dla testu t-studenta (prawostronny) dla próbek niezależnych wynosi: {abs(t_stat)} dla {df} stopni swobody")
print(f"Wartość krytyczna dla danych parametrów wynosi: {cv}, a wartość p: {p}")
if abs(t_stat) <= cv:
    print(f"Pozostaje hipoteza zerowa ponieważ {abs(t_stat)} <= {cv}")
else: 
    print(f"Przyjmujemy hipotezę alternatywną ponieważ {abs(t_stat)} > {cv}")
if p > alpha:
    print("Pozostaje hipoteza zerowa, czyli średnie są równe")
else:
    print("Przyjmujemy hipotezę alternatywną, czyli środki są równe")


# t-student dla prób zależnych
n = len(data1)
d1 = 0
d2 = 0
for i in range(n):
    d1 = d1 + (data1[i] - data2[i]) ** 2
    d2 = d2 + data1[i] - data2[i]

sd = sqrt((d1 - (d2 ** 2 / n)) / (n - 1))
sed = sd / sqrt(n)

t_stat = (mean1 - mean2) / sed
df = n - 1
cv = t.ppf(1.0 - alpha, df)
p = (1 - t.cdf(abs(t_stat), df)) * 2

print("\n\nTest t-studenta dla prób zależnych")
print(f"\nSuma kwadratów różnic wynosi: {d1}, a suma różnic wynosi: {d2}")
print(f"Odchylenie standardowe różnicy między średnimi wynosi: {sd}")
print("\nWartości z obliczeń")
print(f"Wartość dla testu t-studenta (prawostronny) dla próbek zależnych wynosi: {abs(t_stat)} dla {df} stopni swobody")
print(f"Wartość krytyczna dla danych parametrów wynosi: {cv}, a wartość p: {p}")
if abs(t_stat) <= cv:
    print(f"Pozostaje hipoteza zerowa ponieważ {abs(t_stat)} <= {cv}")
else: 
    print(f"Przyjmujemy hipotezę alternatywną ponieważ {abs(t_stat)} > {cv}")
if p > alpha:
    print("Pozostaje hipoteza zerowa, czyli średnie są równe")
else:
    print("Przyjmujemy hipotezę alternatywną, czyli środki są równe")

t_stat, p = ttest_rel(data1, data2)

print("\nWartości z funkcji SciPy")
print(f"Wartość dla testu t-studenta (prawostronny) dla próbek zależnych wynosi: {abs(t_stat)} dla {df} stopni swobody")
print(f"Wartość krytyczna dla danych parametrów wynosi: {cv}, a wartość p: {p}")
if abs(t_stat) <= cv:
    print(f"Pozostaje hipoteza zerowa ponieważ {abs(t_stat)} <= {cv}")
else: 
    print(f"Przyjmujemy hipotezę alternatywną ponieważ {abs(t_stat)} > {cv}")
if p > alpha:
    print("Pozostaje hipoteza zerowa, czyli średnie są równe\n")
else:
    print("Przyjmujemy hipotezę alternatywną, czyli środki są równe\n")