import pandas as pd
import os
from pathlib import Path

    # "count_consumer" считает количество файлов в папке (количество потребителей)
    # нужен для подсчета средней мощности в момент времени
def count_consumer(path_to_files):
    p = path_to_files + '\\Data\\consumer'
    count_consumer = len(os.listdir(p))
    return count_consumer


# объеденение файлов в один файл для последущей его обработки
# функция возвращает массив данных в формате (время вкл, выкл, мощность)
def reading_start_data(path_to_files):
    p = path_to_files + '\\Data\\consumer'
    path = Path(p)
    min_excel_file_size = 100
    df = pd.concat(
        [pd.read_excel(f) 
        for f in path.glob("*.xlsx") 
            if f.stat().st_size >= min_excel_file_size],
        ignore_index=True
    )
        # запись объедененных файлов
    path_to_write = path_to_files + '\\Intermediate_data\\Intermediate_result.xlsx'
    df.to_excel(path_to_write)

    # выгрузка в массив 'd' объедененных данных
    d = pd.DataFrame(df, columns=['t_turn_on', 't_turn_of', 'P'])
    return d
