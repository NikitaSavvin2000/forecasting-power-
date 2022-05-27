import math
import random

import numpy as np

def random_list(N, start, end, medium_values, P_medium):
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
    
    print(x)
    print(P_poisson)

random_list(20, 100, 200, 10, 1000)