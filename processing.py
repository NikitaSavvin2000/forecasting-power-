from functools import reduce
import random
from tkinter import W
import pandas as pd
import numpy as np
import math
from parse import parse_hour_in_min, parse_min_in_hour
import math
import random
import itertools
import numpy as np
from numpy import append, array
import operator

def oneDArray(x):
    return list(itertools.chain(*x))


def processed_data (d,Data_count, t_step_min, count_values):
    a = int(((24*60)/t_step_min) + 1)
    A = [0.0]*a
    t_turn_on = np.array(d['t_turn_on'])
    t_turn_of = np.array(d['t_turn_of'])
    P = np.array(d['P'])
    # проход по массиву длинной P 
    for i in range(len(P)):
        # минуты включения к примеру включен в 1:00 выключен в 2:00 ->
        # -> включен в 60 мин выключен в 120 мин от начала суток
        min_on = parse_hour_in_min(t_turn_on[i])
        min_of = parse_hour_in_min(t_turn_of[i])
        # определяет в какую ячейку списка вписывать начало включения
        number_in_list = math.trunc(min_on/t_step_min)
        # "Y" - число обозначающее сколько промежутков времени задевает включение
        # если интервал равен 30 мин а время включения 13:30 выключения 14:35
        # то задевает 3 промежутка времени по 30 минут
        Y = math.ceil((min_of -min_on)/t_step_min)
        # проверяет краевой случай, если время включения больше времени выключения
        # к примеру включение 16:20 2:00 выключение
        if min_on>=min_of:
            for j in range(len(A)):
                A[j] += P[i]
            on = math.ceil((min_on)/t_step_min)
            of = math.ceil((min_of)/t_step_min)
            for k in range(of, on):
                A[k] -= P[i]
            # записывает данные при стандартном включении выключении, когда min_on < min_of
        for j in range(number_in_list, number_in_list+Y, 1):
            A[j] += P[i]
        
    r = {
        'T' : [x*t_step_min for x in range(len(A))],
        'P' : [p/count_values for p in A],
        'Time' : [parse_min_in_hour(x*t_step_min) for x in range(len(A))],
    }
    result = pd.DataFrame(data = r)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm\\Result.xlsx')
    Data = pd.read_excel(r'C:\\Users\\nsavvin\\Desktop\\programm\\Result.xlsx' )
    return Data

def test(Data_count, t_step_min, N):

    B = []

    t_count_turn_on = np.array(Data_count['t_turn_on'])
    t_count_turn_off = np.array(Data_count['t_turn_off'])

    count = np.array(Data_count['Count'])
    on_list =[]
    off_list = []
    for i in range(len(count)):
        min_on_count = parse_hour_in_min(t_count_turn_on[i])
        min_off_count = parse_hour_in_min(t_count_turn_off[i])
        on = math.ceil((min_on_count)/t_step_min)
        off = math.ceil((min_off_count)/t_step_min)
        if off == 0 and on > 0:
            off = int(((24*60)/t_step_min))
        
        on_list.append(on)
        off_list.append(off)
        
            # записывает данные при стандартном включении выключении, когда min_on < min_of
    r = {
        'time_on' : [q for q in Data_count['t_turn_on']],
        'teme_off' : [w for w in Data_count['t_turn_off']],
        'index_on' : [x for x in on_list],
        'index_off' : [y for y in off_list],
        'Count' : [count for count in Data_count['Count']]
    }
    
    result = pd.DataFrame(data = r)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Count result.xlsx')
    Data_consumer = pd.read_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Count result.xlsx') 
    
    return B

def random_list_interval(N, Data):
    # 6 - количество секунд в условной еденице времени нужно будет автоматически рассчитать
    index_time = []
    format_time = []
    p_list = []
    for r in range(len(Data['Time'])-1):
        start = r * 100
        end = (r+1) * 100
        x = random.sample(range(start, end-1), N)
        x.sort()
        x.append(end)
        index_time.append(x)
        count_sec = [i*6 for i in x]# 6 - количество секунд в условной еденице времени нужно будет автоматически рассчитать
        p = Data['P'][r]
        for i in range(len(count_sec)): 
            hour = math.floor(count_sec[i]/3600)
            minute = math.floor((count_sec[i]-hour*3600)/60)
            sec = (count_sec[i]-hour*3600 - minute* 60)
            time = str(hour) + ':' +  str(minute) + ':' +  str(sec)
            format_time.append(time)
            p_list.append(p)
    index_time_normal =  [x for l in index_time for x in l]
    r ={
        'index_time_normal' : [q for q in index_time_normal],
        'format_time' : [w for w in format_time],
        'p_list' : [e for e in p_list]
    }
    result = pd.DataFrame(data = r)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\random_list_interval.xlsx')


def poisson_all(N):
    p_expected_list = []
    Data_consumer = pd.read_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Count result.xlsx')
    Data_list_interval = pd.read_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\random_list_interval.xlsx')
    index_time_normal = Data_list_interval['index_time_normal']
    format_time = Data_list_interval['format_time']
    p_list = Data_list_interval['p_list']
    index_on_list = Data_consumer['index_on']
    index_off_list = Data_consumer['index_off']
    count_consumer_list = Data_consumer['Count']
    poisson_list =[]
    for i in range(len(count_consumer_list)):
        index_on = index_on_list[i]*N
        index_off = index_off_list[i]*N
        count_consumer_medium = count_consumer_list[i]
        count_index = index_off - index_on
        s = np.random.poisson(count_consumer_medium, count_index).tolist()
        poisson_list.extend(s)
    for q in range(len(poisson_list)):
        p_expected = (p_list[q])*(poisson_list[q])
        p_expected_list.append(p_expected)
    r = {
        'format_time': [t for t in format_time],
        'Time': [q for q in index_time_normal],
        'P_medium' : [e for e in p_list],
        'Count_consumer_random' : [w for w in poisson_list],
        'P_random' : [r for r in p_expected_list]
    }   
    result = pd.DataFrame(data = r)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Poisson P.xlsx')

    

def medium_cout_on_for_interval(): #Среднее количство включений по интервалам времени 
    pass



def min_and_max_poisson(Q, Data):
    Data_min = []
    Data_max = []
    R = list(map(list, zip(*Q))) # транспонирование списка списков
    for i in range(len(R)):
        P_min = min(R[i])
        P_max = max(R[i])
        Data_min.append(P_min)
        Data_max.append(P_max)
    r = {
        'Time' : [time for time in Data['Time']],
        'Min count consumer' : [(Data_min[i]/Data['P'][i])*1000 for i in range(len(Data['P']))],
        'P_min_poisson_kW' : [min for min in Data_min],
        'Max count consumer' : [(Data_max[i]/Data['P'][i])*1000 for i in range(len(Data['P']))],
        'P_max_poisson_kW' : [max for max in Data_max]
    }
    result = pd.DataFrame(data = r)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Intermediate_data\\max_min_poisson.xlsx')
    
    return Data_min, Data_max
