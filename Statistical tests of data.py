import csv
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import shapiro, normaltest, friedmanchisquare, wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(file_path, title):
    temp1 = list()
    temp2 = list()
    temp3 = list()
    r_temp1 = list()
    r_temp2 = list()
    r_temp3 = list()
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        if(title != 'Kontrolna'):
            for row in reader:
                temp1.append(float(row[1]))
                temp1.append(float(row[4]))
                temp2.append(float(row[2]))
                temp2.append(float(row[5]))  
                temp3.append(float(row[3]))
                temp3.append(float(row[6]))
                r_temp1.append(float(row[7]))
                r_temp1.append(float(row[10]))
                r_temp2.append(float(row[8]))
                r_temp2.append(float(row[11]))
                r_temp3.append(float(row[9]))
                r_temp3.append(float(row[12]))
        else:
            for row in reader:
                temp1.append(float(row[1]))
                temp2.append(float(row[2]))
                r_temp1.append(float(row[3]))
    return temp1, temp2, temp3, r_temp1, r_temp2, r_temp3

def pause():
    print('-----------------------------------------------------------------------------------')

def subtraction(temp1, temp2):
    r_temp = list()
    for i in range(len(temp1)):
        r_temp.append(round((temp2[i] - temp1[i]), 2))
    return r_temp

def Repeated_Measures_ANOVA(temp1, temp2, temp3, alpha):
    data = pd.DataFrame({
        'patient': [f'P{i+1}' for i in range(20)],
        'time1': temp1,
        'time2': temp2,
        'time3': temp3
    })
    data_long = pd.melt(data, id_vars=['patient'], value_vars=['time1', 'time2', 'time3'],
                    var_name='time', value_name='temperature')
    model = AnovaRM(data_long, 'temperature', 'patient', within=['time'])
    anova_results = model.fit()
    summary = anova_results.summary().tables[0]
    f_value = summary["F Value"].iloc[0]
    p_value = summary["Pr > F"].iloc[0]
    print(f"{anova_results}")
    print(f"Wartość statystyki F: {f_value:.2f}")
    print(f"Wartość p: {p_value}")
    if p_value < alpha:
        print("Wynik jest istotny statystycznie (p < %s). Możemy odrzucić hipotezę zerową." % alpha)
        print("Oznacza to, że różnice między czasami pomiaru są statystycznie istotne.")
    else:
        print("Wynik nie jest istotny statystycznie (p >= %s). Nie możemy odrzucić hipotezy zerowej." % alpha)
        print("Oznacza to, że różnice między czasami pomiaru nie są statystycznie istotne.")
    pause()

def friedman_chisquare(temp1, temp2, temp3, alpha):
    data = {
    'value': np.concatenate([temp1, temp2, temp3]),
    'label': ['temp1'] * 20 + ['temp2'] * 20 + ['temp3'] * 20,
    'subject': np.tile(np.arange(1, 21), 3)
    }
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index='subject', columns='label', values='value')
    stat, p = friedmanchisquare(df_pivot['temp1'], df_pivot['temp2'], df_pivot['temp3'])
    print('Test Friedmana: statystyka=%.2f, p-wartość=%.8f' % (stat, p))
    if p < alpha:
        print("Wynik jest istotny statystycznie (p < %s). Możemy odrzucić hipotezę zerową." % alpha)
        print("Oznacza to, że różnice między temperaturami w dwóch odstępach czasowych są statystycznie istotne.")
    else:
        print("Wynik nie jest istotny statystycznie (p >= %s). Nie możemy odrzucić hipotezy zerowej." % alpha)
        print("Oznacza to, że różnice między temperaturami w dwóch odstępach czasowych nie są statystycznie istotne.")
    pause()

    groups = ['temp1', 'temp2', 'temp3']
    p_values = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            stat, p = wilcoxon(df_pivot[groups[i]], df_pivot[groups[j]])
            p_values.append(p)
            print('Wilcoxon test pomiędzy "%s" i "%s": p-value=%.8f' % (groups[i], groups[j], p))
    reject, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
    print('Corrected p-values: %.8f, %.8f, %.8f' % (p_corrected[0], p_corrected[1], p_corrected[2]))
    print('Reject null hypothesis:', reject)

def Paired_t_test(time1, time2, alpha):
    data = pd.DataFrame({
    'patient': [f'P{i+1}' for i in range(20)],
    'time1': time1,
    'time2': time2
    })
    time1 = data['time1']
    time2 = data['time2']
    t_stat, p_value = ttest_rel(time1, time2)
    print('t-student')
    print(f"Statystyka t: {t_stat:.4f}")
    print("Wartość p: %.4f" % p_value)
    if p_value < alpha:
        print("Wynik jest istotny statystycznie (p < %s). Możemy odrzucić hipotezę zerową." % alpha)
        print("Oznacza to, że różnice między temperaturami w dwóch odstępach czasowych są statystycznie istotne.")
    else:
        print("Wynik nie jest istotny statystycznie (p >= %s). Nie możemy odrzucić hipotezy zerowej." % alpha)
        print("Oznacza to, że różnice między temperaturami w dwóch odstępach czasowych nie są statystycznie istotne.")
    pause()

def normal_test_1(temp1, temp2, temp3, alpha, title):
    data = {
    'Temperatura [°C]': np.concatenate([temp1, temp2, temp3]),
    'legenda': ['temp1'] * 20 + ['temp2'] * 20 + ['temp3'] * 20
    }
    df = pd.DataFrame(data)
    sns.histplot(data=df, x='Temperatura [°C]', hue='legenda', kde=True, element='step')
    plt.ylabel('Ilość obserwacji')
    plt.show()
    print(f"\n\n{title}")
    pause()
    print('Test normalności:')
    print('- Shapiro-Wilk Test')
    for legenda in df['legenda'].unique():
        __, p1 = shapiro(df[df['legenda'] == legenda]['Temperatura [°C]'])
        if p1 > alpha:
            print("Dla danych '%s' istnieje rozkład normalny (p=%.2f >= %s). Przyjmujemy hipotezę zerową." % (legenda, p1, alpha))
        else:
            print("Dla danych '%s' nie istnieje rozkład normalny (p=%.2f < %s). Możemy odrzucić hipotezę zerową." % (legenda, p1, alpha))
    print('- Agostino Test')
    for legenda in df['legenda'].unique():
        __, p2 = normaltest(df[df['legenda'] == legenda]['Temperatura [°C]'])
        if p2 > alpha:
            print("Dla danych '%s' istnieje rozkład normalny (p=%.2f >= %s). Przyjmujemy hipotezę zerową." % (legenda, p2, alpha))
        else:
            print("Dla danych '%s' nie istnieje rozkład normalny (p=%.2f < %s). Możemy odrzucić hipotezę zerową." % (legenda, p2, alpha))
    pause()

def normal_test_2(temp1, temp2, alpha, title):
    data = {
    'Temperatura [°C]': np.concatenate([temp1, temp2]),
    'legenda': ['temp1'] * 20 + ['temp2'] * 20
    }
    df = pd.DataFrame(data)
    sns.histplot(data=df, x='Temperatura [°C]', hue='legenda', kde=True, element='step')
    plt.ylabel('Ilość obserwacji')
    plt.show()
    print(f"\n\n{title}")
    pause()
    print('Test normalności:')
    print('- Shapiro-Wilk Test')
    for legenda in df['legenda'].unique():
        __, p1 = shapiro(df[df['legenda'] == legenda]['Temperatura [°C]'])
        if p1 > alpha:
            print("Dla danych '%s' istnieje rozkład normalny (p=%.2f >= %s). Przyjmujemy hipotezę zerową." % (legenda, p1, alpha))
        else:
            print("Dla danych '%s' nie istnieje rozkład normalny (p=%.2f < %s). Możemy odrzucić hipotezę zerową." % (legenda, p1, alpha))
    print('- Agostino Test')
    for legenda in df['legenda'].unique():
        __, p2 = normaltest(df[df['legenda'] == legenda]['Temperatura [°C]'])
        if p2 > alpha:
            print("Dla danych '%s' istnieje rozkład normalny (p=%.2f >= %s). Przyjmujemy hipotezę zerową." % (legenda, p2, alpha))
        else:
            print("Dla danych '%s' nie istnieje rozkład normalny (p=%.2f < %s). Możemy odrzucić hipotezę zerową." % (legenda, p2, alpha))
    pause()

def view_1(values1, values2, values3):
    labels = ['Temp1', 'Temp2', 'Temp3']
    x = np.arange(20)
    width = 0.3
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width, values1, width, label=labels[0])
    bar2 = ax.bar(x, values2, width, label=labels[1])
    bar3 = ax.bar(x + width, values3, width, label=labels[2])
    ax.set_xlabel('Osoba badana')
    ax.set_ylabel('Temperatura [°C]')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()
    plt.ylim(25, 35)
    plt.show()

def view_2(values1, values2):
    labels = ['Temp1', 'Temp2', 'Temp3']
    x = np.arange(20)
    width = 0.3
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width, values1, width, label=labels[0])
    bar2 = ax.bar(x, values2, width, label=labels[1])
    ax.set_xlabel('Osoba badana')
    ax.set_ylabel('Temperatura [°C]')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()
    plt.ylim(25, 35)
    plt.show()
                
def graph_1(r_temp1, r_temp2, r_temp3):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    avg_r_temp1 = list()
    avg_r_temp2 = list()
    avg_r_temp3 = list()
    avg_r_temp1_value = sum(r_temp1) / len(r_temp1)
    avg_r_temp2_value = sum(r_temp2) / len(r_temp2)
    avg_r_temp3_value = sum(r_temp3) / len(r_temp3)
    for i in range(len(r_temp1)):
        avg_r_temp1.append(avg_r_temp1_value)
        avg_r_temp2.append(avg_r_temp2_value)
        avg_r_temp3.append(avg_r_temp3_value)
    plt.figure()
    plt.plot(x, r_temp1, 'ro', label='$\Delta$(Temp2 - Temp1)')
    plt.plot(x, r_temp2, 'go', label='$\Delta$(Temp3 - Temp1)')
    plt.plot(x, r_temp3, 'bo', label='$\Delta$(Temp3 - Temp2)')
    plt.plot(x, avg_r_temp1, 'r--', label='Average $\Delta$(Temp2 - Temp1)')
    plt.plot(x, avg_r_temp2, 'g--', label='Average $\Delta$(Temp3 - Temp1)')
    plt.plot(x, avg_r_temp3, 'b--', label='Average $\Delta$(Temp3 - Temp2)')
    print(avg_r_temp1)
    print(avg_r_temp2)
    print(avg_r_temp3)
    plt.xlabel('Osoba badana')
    plt.ylabel('$\Delta$ Temperatura [°C]')
    plt.legend()
    plt.show()

def graph_2(r_temp1):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    avg_r_temp1 = list()
    avg_r_temp1_value = sum(r_temp1) / len(r_temp1)
    for i in range(len(r_temp1)):
        avg_r_temp1.append(avg_r_temp1_value)
    plt.figure()
    plt.plot(x, r_temp1, 'ko', label='$\Delta$(Temp2 - Temp1)')
    plt.plot(x, avg_r_temp1, 'k--', label='Average $\Delta$(Temp2 - Temp1)')
    plt.xlabel('Osoba badana')
    plt.ylabel('$\Delta$ Temperatura [°C]')
    plt.legend()
    plt.show()




# key = ['T_4_5Hz', 'T_50_200Hz', 'U_1MHz', 'U_3MHz', 'Kontrolna']
key = ['U_1MHz', 'U_3MHz', 'Kontrolna']
alpha = 0.05
for i in range(2):
    title = key[i]
    file_path = 'wyniki_%s_M.txt' % title
    temp1, temp2, temp3, r_temp1, r_temp2, r_temp3 = read_data(file_path, title)
    view_1(temp1, temp2, temp3)
    normal_test_1(temp1, temp2, temp3, alpha, title)
    Repeated_Measures_ANOVA(temp1, temp2, temp3, alpha)
    friedman_chisquare(temp1, temp2, temp3, alpha)
    graph_1(r_temp1, r_temp2, r_temp3)


i = 2
title = key[i]
file_path = 'wyniki_%s_M.txt' % title
temp1, temp2, __, r_temp1, __, __ = read_data(file_path, title)
view_2(temp1, temp2)
normal_test_2(temp1, temp2, alpha, title)
Paired_t_test(temp1, temp2, 0.05)
graph_2(r_temp1)