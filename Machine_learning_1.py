from ucimlrepo import fetch_ucirepo
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
import matplotlib.pyplot as plt
import numpy as np
import time



#np.set_printoptions(threshold=np.inf)      # полный вывод массивов



# Пункт 1.2
wholesale_customers = fetch_ucirepo(id=292)
x = wholesale_customers.data.features
y = wholesale_customers.data.targets



# Пункт 1.3
x_train = np.array(x)
y = np.array(y)
x_scaled = preprocessing.StandardScaler().fit_transform(x_train)



# Пункт 2.1
pca = PCA()
X = pca.fit_transform(x_scaled)
eigenvectors = pca.fit_transform(X)
eigenvalues = pca.explained_variance_
print("Собственные векторы:\n",  eigenvectors, "\n")
print("Собственные значения:\n", eigenvalues,  "\n")



# Пункт 2.2
pca = PCA(n_components=2)
X_2 = pca.fit_transform(x_scaled)

plt.figure()
plt.scatter(X_2[:, 0], X_2[:, 1], c=y)

plt.colorbar(label="Regions")
plt.title("Двухмерная карта")
plt.xlabel("X - первый параметр")
plt.ylabel("Y - второй параметр")

plt.show()



# Пункт 2.3
pca = PCA(n_components=3)
X_3 = pca.fit_transform(x_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X_3[:, 0], X_3[:, 1], X_3[:, 2], c=y)

plt.colorbar(scatter, label="Regions")
ax.set( title="Трехмерная карта",
        xlabel="X - первый параметр",
        ylabel="Y - второй параметр",
        zlabel="z - третий параметр")

plt.show()



# Пункт 2.4
n_optimal = 0

for i in range(1, len(eigenvalues)):
    accuracy = sum(eigenvalues[:i]) / sum(eigenvalues) * 100
    print("Количество измерений:", i, "\tПотеря:", 100 - accuracy)
    if accuracy >= 80:
        print("Количество измерений:", i, " -  Оптимально\n")
        n_optimal = i
        break



# Пункт 3
def KNC(X, n, weights, best):
    knc = KNeighborsClassifier(n_neighbors=n, weights=weights)
    loo = LeaveOneOut()

    right, wrong = 0, 0

    start_time = time.time()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knc.fit(X_train, y_train.ravel())
        y_predict = knc.predict(X_test)

        if y_predict == y_test.ravel():
            right += 1
        else:
            wrong += 1
    end_time = time.time()

    if best[0] < right / (right + wrong):
        best[0] = right / (right + wrong)
        best[1] = end_time - start_time
        best[2] = n

    if n < 10: print("Количество соседей:", n, " ", end='')
    else:      print("Количество соседей:", n,  "", end='')

    print("| Размерность:", len(X[0]), "| Вес:", weights, "| Правильно:", right,"| Неправильно:", wrong,
          "| Точность:", f'{right / (right + wrong):.16f}', "| Производительность:", end_time - start_time)



def RNC(X, radius, weights, best):
    rnc = RadiusNeighborsClassifier(radius=radius, weights=weights)
    loo = LeaveOneOut()

    right, wrong = 0, 0

    start_time = time.time()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rnc.fit(X_train, y_train.ravel())
        y_predict = rnc.predict(X_test)

        if y_predict == y_test.ravel():
            right += 1
        else:
            wrong += 1
    end_time = time.time()

    if best[0] < right / (right + wrong):
        best[0] = right / (right + wrong)
        best[1] = end_time - start_time
        best[2] = radius

    if radius < 10: print("Радиус:", radius, " ", end='')
    else:           print("Радиус:", radius,  "", end='')

    print("| Размерность:", len(X[0]), "| Вес:", weights, "| Правильно:", right,"| Неправильно:", wrong,
          "| Точность:", f'{right / (right + wrong):.16f}', "| Производительность:", end_time - start_time)



def NC(X, metric):
    nc = NearestCentroid(metric=metric)
    loo = LeaveOneOut()

    right, wrong = 0, 0

    start_time = time.time()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        nc.fit(X_train, y_train.ravel())
        y_predict = nc.predict(X_test)

        if y_predict == y_test.ravel():
            right += 1
        else:
            wrong += 1
    end_time = time.time()

    print("Размерность:", len(X[0]), "| Метрика:", metric, "| Правильно:", right,"| Неправильно:", wrong,
          "| Точность:", f'{right / (right + wrong):.16f}', "| Производительность:", end_time - start_time)



for weight in ["uniform", "distance"]:
    for component in range(1, len(eigenvalues) + 1):
        pca = PCA(n_components=component)
        X = pca.fit_transform(x_scaled)

        best = [0, 0, 0]

        for n in range(1, 100, 2):
            KNC(X, n, weight, best)

        print("\n\n", "Точность:", best[0], "Производительность:", best[1], "Количество соседей:", best[2], "\n\n")



for weight in ["uniform", "distance"]:
    for component in range(1, len(eigenvalues) + 1):
        pca = PCA(n_components=component)
        X = pca.fit_transform(x_scaled)

        best = [0, 0, 0]

        for radius in range(15, 25):
            RNC(X, radius, weight, best)

        print("\n\n", "Точность:", best[0], "\tПроизводительность:", best[1], "\tРадиус:", best[2], "\n\n")



for metric in ["euclidean", "manhattan"]:
    for component in range(1, len(eigenvalues) + 1):
        pca = PCA(n_components=component)
        X = pca.fit_transform(x_scaled)

        NC(X, metric)

    print("\n\n")
