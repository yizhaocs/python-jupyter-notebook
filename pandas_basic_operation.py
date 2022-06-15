# -*- coding: utf-8 -*-
"""Pandas的基本操作.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IolW9RUrR2lGOeoFZQfq67jgxy6NYBgZ
"""

import pandas as pd


def pandas_shape():
    input_data = 'Resources/host_health.csv'
    df = pd.read_csv(input_data, lineterminator='\n')
    print(f"row count:{df.shape[0]}, column count:{df.shape[1]}")  # row count:3549, column count:6


'''
找出所有unique
'''


def pandas_unique():
    input_data = 'Resources/host_health.csv'
    df = pd.read_csv(input_data, lineterminator='\n')
    unique_host_name_list = list(df['Host Name'].unique())
    # 21
    print(len(unique_host_name_list))
    # ['FSM-GFU-Window2012R2-WIN2012R2-172-30-56-123', 'ussvnplesx54.fortinet-us.com', 'ussvnplesx58.fortinet-us.com', 'ussvnplesx56.fortinet-us.com', 'ussvnplesx59.fortinet-us.com', 'FSM-GFU1-Window2022-WIN2022-172-30-56-125', 'ussvnplesx51.fortinet-us.com', 'ussvnplesx53.fortinet-us.com', 'ussvnplesx52.fortinet-us.com', 'ussvnplesx57.fortinet-us.com', 'ussvnplesx55.fortinet-us.com', 'FSM-GFU-Window10-WIN10-172-30-56-127', 'FSM-GFU-Window2016-WIN2016test-172-30-56-126.gfu.com', 'WIN-QA-RDP-230', 'FSM-CPWANG-WIND2019-172-30-56-214', 'WIN-2R27C8ADMT3.fortisiem-lab.net', 'FSM-RPRAJUDHA-CENTOS8-NFS217-172.30.57.217', 'fsm-ntwk-wlan01', 'it-rle-oradb-57141', 'FSM-TTRINH-CENTOS8-KAFKA-172.30.56.34', 'HOST-172.30.56.104']
    print(unique_host_name_list)
    # row count:3549, column count:6
    print(f"row count:{df.shape[0]}, column count:{df.shape[1]}")

    for host_name in unique_host_name_list:
        same_host_name_group_df = df[df['Host Name'] == host_name].reset_index(drop=True)
        print(f"host_name:{host_name}")
        print(f"row count:{same_host_name_group_df.shape[0]}, column count:{same_host_name_group_df.shape[1]}")
        print(same_host_name_group_df['Host Name'])


'''
group by
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
'''
def pandas_groupby():
    df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                       'Speed': [380., 370., 24., 26.]})
    print(df)
    df_groupby = df.groupby(['Animal'])

    print(df_groupby.keys)  # ['Animal']
    print(len(df_groupby.groups))  # 2

    print(df_groupby.mean())
    print(df_groupby.sum())


def pandas_append_df():
    input_data = 'Resources/housing.csv'
    df = pd.read_csv(input_data, lineterminator='\n')

    for i in range(0, 10):
        df = df.append(df)

    # 21
    print(df.shape)
    # ['FSM-GFU-Window2012R2-WIN2012R2-172-30-56-123', 'ussvnplesx54.fortinet-us.com', 'ussvnplesx58.fortinet-us.com', 'ussvnplesx56.fortinet-us.com', 'ussvnplesx59.fortinet-us.com', 'FSM-GFU1-Window2022-WIN2022-172-30-56-125', 'ussvnplesx51.fortinet-us.com', 'ussvnplesx53.fortinet-us.com', 'ussvnplesx52.fortinet-us.com', 'ussvnplesx57.fortinet-us.com', 'ussvnplesx55.fortinet-us.com', 'FSM-GFU-Window10-WIN10-172-30-56-127', 'FSM-GFU-Window2016-WIN2016test-172-30-56-126.gfu.com', 'WIN-QA-RDP-230', 'FSM-CPWANG-WIND2019-172-30-56-214', 'WIN-2R27C8ADMT3.fortisiem-lab.net', 'FSM-RPRAJUDHA-CENTOS8-NFS217-172.30.57.217', 'fsm-ntwk-wlan01', 'it-rle-oradb-57141', 'FSM-TTRINH-CENTOS8-KAFKA-172.30.56.34', 'HOST-172.30.56.104']
    df.to_csv('/Users/yzhao/Downloads/housing_10.csv', index=False)


if __name__ == '__main__':
    # pandas_unique()
    pandas_groupby()
    # pandas_shape()
    # pandas_append_df()
