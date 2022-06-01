import itertools
import pandas as pd
from processing import processed_data, min_max_medium_poisson, poisson_all,random_list_interval, iteration, write_out_data
from reading import reading_start_data, count_consumer
from graphic import graf_gistogram_medium, graf_gistogram_poison, graf_gistogram_min_max_poison
import numpy as np
import matplotlib.pyplot as plt

path_to_files = 'C:\\Users\\nsavvin\\Desktop\\programm2'


try:
    t_step_min = int(input('Введите шаг времени в минутах - '))
    count_values = int(input('Введите количество дроблений промежутка - '))
    I =  int(input('Введите количество итераций - '))
except TypeError:
    print('Введите корректно минуты без дробных чисел и букв')

Data_count = pd.read_excel('C:\\Users\\nsavvin\\Desktop\\programm2\\count.xlsx')
d = reading_start_data(path_to_files)

count_files = count_consumer(path_to_files)

D = processed_data(d, Data_count, t_step_min, count_files)

#Q = data_poisson(D, medium_values, count_values)



#graf_gistogram_poison(D, data_poisson, time_index_interval, t_step_min)
#Data_min, Data_max = min_and_max_poisson(Q, D)
#graf_gistogram_min_max_poison(D, Data_min, Data_max, t_step_min)

random_list_interval(count_values-1, D)
#plt.show()

N = count_values
#time_index_interval, all_poisson_destribution = poisson_all(N, I)
poisson_all(N)
t, x, y = iteration(N, I)
min_poisson_destribution, max_poisson_destribution, medium_poisson_destribution = min_max_medium_poisson(y)
graf_gistogram_poison(y, x)
graf_gistogram_min_max_poison(min_poisson_destribution, max_poisson_destribution, medium_poisson_destribution, x)
print(x)
write_out_data(
    t, x, y,
    min_poisson_destribution, 
    max_poisson_destribution, 
    medium_poisson_destribution,
    )

#plt.show()
