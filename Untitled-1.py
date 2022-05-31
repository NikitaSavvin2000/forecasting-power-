#!usr/bin/env python
#_*_ coding:utf-8 _*_
 
"""
 title: python реализует алгоритм совместной фильтрации на основе пользователя и содержимого
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from math import sqrt
 
#Файл # u.data содержит полный набор данных.
#u_data_path="C:\\Users\\lenovo\\Desktop\\ml-100k\\"
header = ['user_id', 'item_id']
df = pd.read_csv(r'C:\\Users\\nsavvin\Downloads\\123.csv', sep=';')
print(df.head(5))
print(len(df))
# Обратите внимание на первые две строки данных. Далее посчитаем общее количество пользователей и фильмов.
n_users = df.user_id.unique().shape[0]  #unique () - дедупликация. Количество строк в shape [0]
n_items = df.item_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
train_data, test_data = cv.train_test_split(df, test_size=0.25)
 
#Create two user-item matrices, one for training and another for testing
# Разница между train_data и test_data
train_data_matrix = np.zeros((n_users, n_items))
print(train_data_matrix.shape)
for line in train_data.itertuples():
    train_data_matrix[int(line[0])-1, int(line[1])-1] = int(line[0])
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[0]-1, line[1]-1] = line[0]
# Вы можете использовать функцию pairwise_distances программы sklearn для вычисления косинусного сходства. Обратите внимание: поскольку все оценки положительные, выходное значение должно быть от 0 до 1.
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
# Матрица транспонирования для достижения схожести темы
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
 
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')
 
# Существует много индикаторов оценки, но одним из самых популярных индикаторов для оценки точности прогнозов является среднеквадратичная ошибка (RMSE).

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()#nonzero (a) возвращает индекс элемента, значение которого не равно нулю в массиве a, что эквивалентно извлечению разреженной матрицы
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
 
print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))