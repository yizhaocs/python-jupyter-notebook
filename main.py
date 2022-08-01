import pandas as pd
import datetime
if __name__ == '__main__':
    # input_data = 'Resources/internet_traffic_raw_data.csv'
    # df = pd.read_csv(input_data, lineterminator='\n')
    # dt1 = datetime.datetime.strptime(df['_time'][0], '%Y-%m-%dT%H:%M:%S')
    # dt2 = datetime.datetime.strptime(df['_time'][1], '%Y-%m-%dT%H:%M:%S')
    #
    dt1 = datetime.datetime.strptime('2015-03-02 00:00:00', '%Y-%m-%d %H:%M:%S')
    dt2 = datetime.datetime.strptime('2015-03-02 02:00:00', '%Y-%m-%d %H:%M:%S')
    tdelta = dt2 - dt1
    print(tdelta)
    print(type(tdelta))

    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq=tdelta)
    print(dt_rng)