import pandas as pd

'''
Reference：
https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values


找出所有rows with Host Name == ''ussvnplesx54.fortinet-us.com''
'''


input_data = 'Resources/host_health.csv'
df = pd.read_csv(input_data, lineterminator='\n')
'''
一个condition
'''
filter_result_single_condition = df.loc[df['Host Name'] == 'ussvnplesx54.fortinet-us.com']
print(filter_result_single_condition)

'''
多个conditions
'''
filter_result_multiple_conditions = df.loc[(df['Host Name'] == 'ussvnplesx54.fortinet-us.com') & (df['AVG(CPU Util)'] > 59)]
print(filter_result_multiple_conditions)