import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from itertools import combinations
from datetime import datetime
import time


class OCC:
    def __init__(self, max_clusters: int):
        self.max_clusters = max_clusters

    def _indexToExample(self):
        pass

    def _delete_rows(self, delete_array, delete_index):
        pair = delete_array[delete_index][0:2]
        delete_array = delete_array[delete_array[:, 0] != (pair[0] and pair[1])]
        delete_array = delete_array[delete_array[:, 1] != (pair[0] and pair[1])]
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

    def _validate_length(self, data):
        if len(data) % 2 == 0:
            pass
        else:
            raise ValueError(
                "Length of input array is assumed to be an even number, "
                "otherwise some of the objects are regarded twice."
            )

    def fit(self, data):
        self._validate_length(data)
        if isinstance(data, pd.DataFrame):
            numpy_data = np.array(data)
        else:
            numpy_data = data
        print("Расчет расстояния между примерами...\n")
        indexes = np.array(list(combinations(range(len(data)), 2)))
        indexes_list_length = int(len(data) * (len(data) - 1) / 2)  # Количество комбинаций
        distances_array = np.zeros((indexes_list_length, 3))
        distances_array[:, 0], distances_array[:, 1] = indexes[:, 0], indexes[:, 1]
        distances_array[:, 2] = np.linalg.norm(numpy_data[indexes[:, 0]] - numpy_data[indexes[:, 1]], axis=1)
        print(distances_array)
        print("Разделение на массивы А и В...\n")
        # Разделяем на массивы А и В
        A_examples = np.array([], dtype=int)
        B_examples = np.array([], dtype=int)
        A_indexes = np.array([], dtype=int)
        B_indexes = np.array([], dtype=int)
        start_time = datetime.now()
        while len(distances_array) != 0:
            min_index = np.argmin(distances_array, axis=0)[2]
            A_indexes = np.append(A_indexes, distances_array[min_index][0])
            B_indexes = np.append(B_indexes, distances_array[min_index][1])
            distances_array = self._delete_rows(distances_array, min_index)

            print(len(distances_array))
        print(datetime.now() - start_time)
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
