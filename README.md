# Основы машинного обучения | Лабораторная работа 1 | Вариант "wholesale customers"

В ходе лабораторной работы необходимо:
- 1.1. Найти заданный датасет и описать его в отчете;
- 1.2. Импортировать датасет в "python" в том формате, с которым работает "sklearn";
- 1.3. Нормализовать датасет.

- 2.1. Заполнить таблицу с собственными векторами и собственными значениями;
- 2.2. Снизить размерность датасета до 2 и вывести двухмерную карту датасета;
- 2.3. Снизить размерность датасета до 3 и вывести трехмерную карту датасета;
- 2.4. Снизить размерность датасета до некоторого количества измерений, которое покажется оптимальным исходя из собственных значений.

- 3   Оценить работу метода ближайших соседей на изначальном датасете в режиме "Leave-One-Out" (один из экземпляров датасета классифицируется на основе остальных). Посчитать правильно и неправильно классифицированных соседей и время выполнения классификации. Сравнить точность и производительность метода для разных настроек:
    - Количество ближайших соседей (1,3,5,7,9,...);
    - Использование весов (uniform – равные весы, distance – веса зависят от расстояния);
    - Размерность пространств (исходное, либо разные пространства, полученные методом главных компонент);
    - Nearest neighbors vs Radius Neighbors vs Nearest Centroid (с разными параметрами).
Итог: Результаты занести в таблицы.
