import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from itertools import combinations


class OCC:
    def __init__(self, max_clusters: int):
        self.max_clusters = max_clusters

    def _delete_rows(self, delete_array, delete_index):
        pair = delete_array[delete_index][0:2]
        delete_array = delete_array[delete_array[:, 0] != pair[0]]
        delete_array = delete_array[delete_array[:, 0] != pair[1]]
        delete_array = delete_array[delete_array[:, 1] != pair[0]]
        delete_array = delete_array[delete_array[:, 1] != pair[1]]
        return delete_array

    def _count_clusters(self, A, B, row_cluster_n, column_cluster_n):
        counter = 0
        for i in A.loc[A[0] == row_cluster_n][0].index:
            if i in B.loc[B[0] == column_cluster_n][0].index:
                counter = counter + 1
        return counter

    def _print_results(self, res, number_of_clusters):
        print(f'Результаты кластеризации для {number_of_clusters} кластеров:\n')
        for i in range(number_of_clusters):
            print(f'Кластер {i}: совпало {res[i][i]}/{res[i].sum()} точек.')
        print('\n')

    def train(self, data):
        numpy_data = data
        if isinstance(data, pd.DataFrame):
            numpy_data = np.array(data)
        print("Расчет расстояния между примерами...\n")
        # Считаем расстояния между примерами
        indexes = np.array(list(combinations(range(len(data)), 2)))
        indexes_list_length = int(len(data) * (len(data) - 1) / 2)  # Количество комбинаций
        distances_array = np.zeros((indexes_list_length, 3))
        distances_array[:, 0], distances_array[:, 1] = indexes[:, 0], indexes[:, 1]
        distances_array[:, 2] = np.linalg.norm(numpy_data[indexes[:, 0]] - numpy_data[indexes[:, 1]], axis=1)
        print(distances_array)
        print("Разделение на массивы А и В...\n")
        # Разделяем на массивы А и В
        A = np.array([], dtype=int)
        B = np.array([], dtype=int)
        A_index = np.array([], dtype=int)
        B_index = np.array([], dtype=int)
        dipoles = np.array([])
        while len(array) != 0:
            min_index = np.argmin(array, axis=0)[2]
            A_index = np.append(A_index, array[min_index][0])
            B_index = np.append(B_index, array[min_index][1])
            array = self._delete_rows(array, min_index)
            print(len(array))

        A_index = np.int64(A_index)
        B_index = np.int64(B_index)

        # Переходим от индексов к примерам
        for i in range(len(A_index)):
            A = np.append(A, data[A_index[i]])
            B = np.append(B, data[B_index[i]])
        A = np.reshape(A, (-1, data.shape[1]))
        B = np.reshape(B, (-1, data.shape[1]))

        print('Кластеризация...\n')
        # Кластеризация
        for num_clusters in range(2, self.max_clusters + 1):
            clusterizator_A = KMeans(n_clusters=num_clusters, n_init=100, max_iter=1500, random_state=seed)
            predicted_A = pd.DataFrame(clusterizator_A.fit_predict(A))
            predicted_B = pd.DataFrame(clusterizator_A.predict(B))

            res_array = np.zeros((num_clusters, num_clusters), dtype=int)
            for i in range(num_clusters):
                for j in range(num_clusters):
                    res_array[i][j] = self._count_clusters(predicted_A, predicted_B, i, j)
            self._print_results(res_array, num_clusters)
        return clusterizator_A
