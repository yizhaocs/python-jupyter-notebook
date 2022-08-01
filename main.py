import pandas as pd
import datetime

from dateutil.relativedelta import relativedelta

if __name__ == '__main__':
    input_data = 'Resources/internet_traffic_raw_data.csv'
    df = pd.read_csv(input_data, lineterminator='\n')
    first_datetime = datetime.datetime.strptime(df['_time'][0], '%Y-%m-%d %H:%M:%S')
    second_datetime = datetime.datetime.strptime(df['_time'][1], '%Y-%m-%d %H:%M:%S')

    tdelta = second_datetime - first_datetime
    print(f'tdelta:{tdelta}')
    print(f'type(tdelta):{type(tdelta)}')

    last_datetime = datetime.datetime.strptime(df['_time'][len(df) - 1], '%Y-%m-%d %H:%M:%S')
    print(f'last_datetime:{last_datetime}') # last_datetime:2005-07-28 13:00:00

    tdelta_str = str(tdelta)

    future_datetime = None
    if 'days' in tdelta_str:
        future_datetime = last_datetime + relativedelta(days=int(tdelta_str.partition('days')[0]))
    else:
        future_datetime = last_datetime + relativedelta(hours=int(tdelta_str.partition(':')[0]))

    print(f'future_datetime:{future_datetime}')
    # dt_rng = pd.date_range(last_datetime, , freq=tdelta)
    # print(dt_rng)