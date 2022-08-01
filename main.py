import pandas as pd
import datetime
if __name__ == '__main__':
    input_data = 'Resources/internet_traffic_raw_data.csv'
    df = pd.read_csv(input_data, lineterminator='\n')
    first_datetime = datetime.datetime.strptime(df['_time'][0], '%Y-%m-%dT%H:%M:%S')
    second_datetime = datetime.datetime.strptime(df['_time'][1], '%Y-%m-%dT%H:%M:%S')

    first_datetime = datetime.datetime.strptime('2015-03-02 00:00:00', '%Y-%m-%d %H:%M:%S')
    second_datetime = datetime.datetime.strptime('2016-03-02 02:00:00', '%Y-%m-%d %H:%M:%S')
    tdelta = second_datetime - first_datetime
    print(f'tdelta:{tdelta}')
    print(f'type(tdelta):{type(tdelta)}')

    last_datetime = datetime.datetime.strptime(df['_time'][len(df) - 1], '%Y-%m-%dT%H:%M:%S')
    print(f'last_datetime:{last_datetime}')
    dt_rng = pd.date_range('2015-03-02 00:00:00', last_datetime, freq=tdelta)
    print(dt_rng)