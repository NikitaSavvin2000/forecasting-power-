from audioop import avg
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
from numpy import append, array, mean
import operator
import  openpyxl as ox

wb=ox.Workbook()

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
    Data_consumer = pd.read_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\Count result.xlsx')
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
    return poisson_list


def iteration(N, I):
    iteration_list_poisson = []
    Data_list_interval = pd.read_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\random_list_interval.xlsx')
    index_time_normal = Data_list_interval['index_time_normal']
    format_time = Data_list_interval['format_time']
    p_list = Data_list_interval['p_list']
    i = 0
    poisson_list_p_itteration = []
    while i <= I:
        i+=1
        iteration_poisson = poisson_all(N)
        iteration_list_poisson.append(iteration_poisson)
    for i in range(I):
        poisson_list_p = []
        for x in range(len(p_list)):
            p = (iteration_list_poisson[i][x]*p_list[x])
            poisson_list_p.append(p)
        poisson_list_p_itteration.append(poisson_list_p)
    df = pd.DataFrame(poisson_list_p_itteration)
    dft = df.T
    dft.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\random_list_interval_poisson.xlsx')
    return format_time, index_time_normal, poisson_list_p_itteration




def min_max_medium_poisson(iteration_list_poisson):
    Data_min = []
    Data_max = []
    Data_medium = []
    R = list(map(list, zip(*iteration_list_poisson))) # транспонирование списка списков
    for i in range(len(R)):
        P_min = min(R[i])
        P_max = max(R[i])
        P_medium = mean(R[i])
        Data_min.append(P_min)
        Data_max.append(P_max)
        Data_medium.append(P_medium)
    return Data_min, Data_max, Data_medium

def write_out_data(
    format_time,
    index_time_normal,
    poisson_list_p_itteration, 
    min_poisson_destribution, 
    max_poisson_destribution, 
    medium_poisson_destribution,
    ):
    name_column_list = []
    
    for i in range(len(poisson_list_p_itteration)):
        name_column = 'P_poissin_' + str(i+1)
        name_column_list.append(name_column)
    name_column_dict = dict(zip(name_column_list, poisson_list_p_itteration))
    any_result = {
        'Index_time' : index_time_normal,
        'Time' : format_time,
        'Min_poisson': min_poisson_destribution,
        'Max_poisson': max_poisson_destribution,
        'Medium_poisson': medium_poisson_destribution,
    }
    any_result.update(name_column_dict)
    result = pd.DataFrame(any_result)
    result.to_excel(r'C:\\Users\\nsavvin\Desktop\\programm2\\test_result.xlsx')
    
