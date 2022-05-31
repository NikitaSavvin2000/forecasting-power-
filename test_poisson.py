import random
import pandas as pd
import numpy as np
import math
from parse import parse_hour_in_min, parse_min_in_hour
import math
import random
import numpy as np
result_x = []

def random_list_interval(N, start, end, medium_values, P_medium):
    x = random.sample(range(start, end), N)
    x.sort()
    poisson = np.random.poisson(medium_values, N+1)
    x.append(end)
    count_sec = [i*6 for i in x]
    time = []
    for i in range(len(count_sec)):
        format_time = []
        hour = math.floor(count_sec[i]/3600)
        minute = math.floor((count_sec[i]-hour*3600)/60)
        sec = (count_sec[i]-hour*3600 - minute* 60)
        format_time.append(hour)
        format_time.append(minute)
        format_time.append(sec)
        time.append(format_time)
        P_poisson = [poisson[k]*P_medium for k in range(len(poisson))]
    return x, time, P_poisson

for i in range(3):
    start = i * 100
    end = (i+1) * 100
    N = 4
    medium_values = 10
    P_medium =1000
    x, time, P_poisson = random_list_interval(N, start, end, medium_values, P_medium)
    result_x.append(time)
print(result_x)