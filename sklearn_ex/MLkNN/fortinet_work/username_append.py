import pandas as pd
import numpy as np

df = pd.read_csv('data/report1666743279291_with_incident_title.csv')
# df2 = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/sklearn_ex/MLkNN/internet_example/data/so_dataset_2_tags.csv')
df2 = pd.DataFrame(np.random.randint(0, 2, size=(14709, 4)), columns=list('ABCD'))
df = pd.concat([df, df2[['A', 'B', 'C', 'D']]], axis=1)
df.rename(columns={'A': 'user_A', 'B': 'user_B', 'C': 'user_C', 'D': 'user_D'}, inplace=True)
df.to_csv('/Users/yzhao/Downloads/report1666743279291_with_incident_title_with_username.csv', index=False)
