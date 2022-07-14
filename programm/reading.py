import pandas as pd
import os
from pathlib import Path
from path import path_intermediate_result
    # 'count_consumer' считает количество файлов в папке (количество потребителей)
    # нужен для подсчета средней мощности в момент времени
def count_consumer(path_to_files):
    count_consumer = len(os.listdir(path_to_files))
    return count_consumer


# объеденение файлов в один файл для последущей его обработки
# функция возвращает массив данных в формате (время вкл, выкл, мощность)
def reading_start_data(path_to_files):
    path = Path(path_to_files)
    min_excel_file_size = 100
    df = pd.concat(
        [pd.read_excel(f) 
        for f in path.glob('*.xlsx') 
            if f.stat().st_size >= min_excel_file_size],
        ignore_index=True
    )
        # запись объедененных файлов
    df.to_excel(path_intermediate_result)

    # выгрузка в массив 'd' объедененных данных
    count_consumer = len(os.listdir(path_to_files))
    d = pd.DataFrame(df, columns=['t_turn_on', 't_turn_of', 'P'])
    return d, count_consumer
