import pandas as pd
import datetime

from dateutil.relativedelta import relativedelta

if __name__ == '__main__':
    input_data = 'Resources/internet_traffic_raw_data.csv'
    df = pd.read_csv(input_data, lineterminator='\n')
    first_datetime = datetime.datetime.strptime(df['_time'][0], '%Y-%m-%d %H:%M:%S')
    second_datetime = datetime.datetime.strptime(df['_time'][1], '%Y-%m-%d %H:%M:%S')
    print(f'first_datetime:{first_datetime}') # first_datetime:2005-06-07 07:00:00
    print(f'second_datetime:{second_datetime}') # second_datetime:2005-06-07 09:00:00
    second_to_first_diff = second_datetime - first_datetime
    print(f'second_datetime - first_datetime:{second_to_first_diff}') # second_datetime - first_datetime:2:00:00

    last_datetime = datetime.datetime.strptime(df['_time'][len(df) - 1], '%Y-%m-%d %H:%M:%S')
    print(f'last_datetime:{last_datetime}') # last_datetime:2005-07-28 13:00:00
    last_to_first_diff = last_datetime - first_datetime
    print(f'last_datetime - first_datetime:{last_to_first_diff}') # last_datetime - first_datetime:51 days, 6:00:00


    last_to_first_diff_str = str(last_to_first_diff)

    future_datetime = None
    if 'days' in last_to_first_diff_str:
        future_datetime = last_datetime + relativedelta(days=int(last_to_first_diff_str.partition('days')[0]))
    else:
        future_datetime = last_datetime + relativedelta(hours=int(last_to_first_diff_str.partition(':')[0]))

    print(f'future_datetime:{future_datetime}') # future_datetime:2005-09-17 13:00:00

    # DatetimeIndex(['2005-07-28 13:00:00', '2005-07-28 15:00:00',
    #                '2005-07-28 17:00:00', '2005-07-28 19:00:00',
    #                '2005-07-28 21:00:00', '2005-07-28 23:00:00',
    #                '2005-07-29 01:00:00', '2005-07-29 03:00:00',
    #                '2005-07-29 05:00:00', '2005-07-29 07:00:00',
    #                ...
    #                '2005-09-16 19:00:00', '2005-09-16 21:00:00',
    #                '2005-09-16 23:00:00', '2005-09-17 01:00:00',
    #                '2005-09-17 03:00:00', '2005-09-17 05:00:00',
    #                '2005-09-17 07:00:00', '2005-09-17 09:00:00',
    #                '2005-09-17 11:00:00', '2005-09-17 13:00:00'],
    #               dtype='datetime64[ns]', length=613, freq='2H')
    furture_datetime_fill_in_with_frequency = pd.date_range(last_datetime, future_datetime, freq=second_to_first_diff)
    print(furture_datetime_fill_in_with_frequency)