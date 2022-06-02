import math

    # "parse_hour_in_min" парсит формат часов в минуты пример :
    # 2:30 -> 150 минут суток
def parse_hour_in_min(time):
    time = str(time)
    hour_and_min = time.split(':')
    time_min = int(hour_and_min[0])*60 + int(hour_and_min[1])
    return time_min

# "parse_min_in_hour" парсит минуты суток в часы, нужен для более удобной визуализации времени
def parse_min_in_hour(time_min):
    hour = math.floor(time_min/60)
    minute = time_min - (hour*60)
    time = str(hour) + ':' + str(minute)
    if minute == 0:
        time = str(hour) + ':00'
    if hour == 24:
        time = '00:00'
    return time