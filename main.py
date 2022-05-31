import itertools
import pandas as pd
from processing import processed_data, min_and_max_poisson, poisson_all, test, random_list_interval
from reading import reading_start_data, count_consumer
from graphic import graf_gistogram_medium, graf_gistogram_poison, graf_gistogram_min_max_poison
import numpy as np
import matplotlib.pyplot as plt

path_to_files = 'C:\\Users\\nsavvin\\Desktop\\programm2'


try:
    t_step_min = int(input('Введите шаг времени в минутах - '))
    medium_values = int(input('Введите среднее значение включений - '))
    count_values = int(input('Введите количество дроблений промежутка - '))
except TypeError:
    print('Введите корректно минуты без дробных чисел и букв')

Data_count = pd.read_excel('C:\\Users\\nsavvin\\Desktop\\programm2\\count.xlsx')
d = reading_start_data(path_to_files)

count_files = count_consumer(path_to_files)

D = processed_data(d, Data_count, t_step_min, count_files)

#Q = data_poisson(D, medium_values, count_values)


graf_gistogram_medium(D, t_step_min, count_files)
#graf_gistogram_poison(D, Q, t_step_min)
#Data_min, Data_max = min_and_max_poisson(Q, D)
#graf_gistogram_min_max_poison(D, Data_min, Data_max, t_step_min)

random_list_interval(count_values-1, D)
#plt.show()

b = test(Data_count, t_step_min, count_values)
N = count_values
poisson_all(N)

#print(z)
# закончил на том что по Data_min, Data_max нужно сделать апроксимацию, а так-же нужно в функции data_poisson задатть данные  с экрана
