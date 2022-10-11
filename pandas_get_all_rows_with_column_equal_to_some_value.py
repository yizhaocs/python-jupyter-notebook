import pandas as pd

'''
Reference：
https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values


找出所有rows with Host Name == ''ussvnplesx54.fortinet-us.com''
'''


input_data = 'Resources/host_health.csv'
df = pd.read_csv(input_data, lineterminator='\n')
filter_result = df.loc[df['Host Name'] == 'ussvnplesx54.fortinet-us.com']
print(filter_result)