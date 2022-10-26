
import pandas as pd

df = pd.read_csv('data/report1666743279291_with_incident_title.csv')
df2 = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/sklearn_ex/MLkNN/internet_example/data/so_dataset_2_tags.csv')
df = pd.concat([df, df2[['mysql', 'python', 'php']]], axis=1)
df.rename(columns={'mysql': 'user_A', 'python': 'user_B', 'php': 'user_C'}, inplace=True)
df.to_csv('/Users/yzhao/Downloads/report1666743279291_with_incident_title_with_username.csv', index=False)