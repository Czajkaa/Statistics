import pandas as pd
from sklearn.metrics import f1_score

# Tablica pomyłek
def confusion_matrix(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return unique, matrix

def print_confusion_matrix(unique, matrix):
    unique_list = list(unique)
    matrix_value = list()
    for x in range(len(matrix)):
        row_value = list()
        for y in range(len(unique)):
            value = matrix[x][y]
            row_value.append(value)
        matrix_value.append(row_value)
    df = pd.DataFrame(matrix_value, index=unique_list, columns=unique_list)
    print(df)

# F1 score
def print_F1_score(actual, predicted, unique):
    unique_list = list(unique)
    f1_avg = f1_score(actual, predicted, average='weighted')
    f1 = f1_score(actual, predicted, average=None)
    print('Średni F1 score: %.3f ->' % f1_avg)
    for x in range(len(unique)):
        print('Etykieta %s: %.3f' % (unique_list[x], f1[x]))


# Implementacja
actual =    [0,0,0,0,0,1,1,1,1,1,33,1,2,2,0,0,0,0,0,1,1,1,1,1,33,1,2,2,0,0,0,0,0,1,1,1,1,1,33,1,2,2]
predicted = [0,1,1,0,0,1,0,1,1,0,33,33,2,1,0,0,0,0,0,1,1,1,1,1,33,1,2,2,0,0,0,0,0,1,1,1,1,1,33,1,2,2]
unique, matrix = confusion_matrix(actual, predicted)


print_confusion_matrix(unique, matrix)
print()
print_F1_score(actual, predicted, unique)
print()
print(actual)