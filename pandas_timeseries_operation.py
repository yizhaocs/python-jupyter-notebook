import pandas as pd
import numpy as np
import datetime


def ex1():
    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq='H')
    print(dt_rng)


def ex2():
    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq='H')
    print(dt_rng)


def ex3():
    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq='2H')
    print(dt_rng)


def ex4():
    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-02 3:00:00', freq='15min')
    print(dt_rng)

    dt1 = datetime.datetime.strptime('2015-03-02 00:00:00', '%Y-%m-%d %H:%M:%S')
    dt2 = datetime.datetime.strptime('2015-03-02 02:00:00', '%Y-%m-%d %H:%M:%S')
    tdelta = dt2 - dt1
    print(tdelta)
    print(type(tdelta))

    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq=tdelta)
    print(dt_rng)


def ex5():
    input_data = 'Resources/internet_traffic_raw_data.csv'
    df = pd.read_csv(input_data, lineterminator='\n')
    dt1 = datetime.datetime.strptime(df['_time'][0], '%Y-%m-%dT%H:%M:%S')
    dt2 = datetime.datetime.strptime(df['_time'][1], '%Y-%m-%dT%H:%M:%S')

    tdelta = dt2 - dt1
    print(tdelta)
    print(type(tdelta))

    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq=tdelta)
    print(dt_rng)


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
