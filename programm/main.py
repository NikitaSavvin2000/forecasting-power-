import pandas as pd
from processing import (
    processed_data, 
    min_max_medium_poisson, 
    poisson_all,random_list_interval, 
    iteration, write_out_data, 
    write_count_consumer
    )
from reading import reading_start_data, count_consumer
from graphic import (
    graf_gistogram_medium,
    graf_gistogram_min_max_medium_poison
)
import matplotlib.pyplot as plt
from path import *


try:
    t_step_min = int(input('Введите шаг времени в минутах - '))
    count_values = int(input('Введите количество дроблений промежутка - '))
    I =  int(input('Введите количество итераций - '))
except TypeError:
    print('Введите корректно минуты без дробных чисел и букв')

Data_count = pd.read_excel(path_any_data)

d, count_files = reading_start_data(path_consumer)

D = processed_data(d, t_step_min, count_files)

u = write_count_consumer(Data_count, t_step_min)

random_list_interval(count_values-1, D, t_step_min)

N = count_values

poisson_list = poisson_all(N)

t, x, y = iteration(N, I)

(
    min_poisson_destribution,
    max_poisson_destribution,
    medium_poisson_destribution
)= min_max_medium_poisson(y)


graf_gistogram_min_max_medium_poison(
    D, min_poisson_destribution, 
    max_poisson_destribution, 
    medium_poisson_destribution, 
    x, t_step_min
)

graf_gistogram_medium(D, t_step_min, count_files)

write_out_data(
    t, x, y,
    min_poisson_destribution, 
    max_poisson_destribution, 
    medium_poisson_destribution
)

plt.show()
