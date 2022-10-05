import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


feature_columns = ["Event Name"]

df = pd.read_csv("../Resources/report1664845422878.csv")
print(f'df.head():{df.head()}')
print(f'df.tail():{df.tail()}')
print(f'df.dtypes:{df.dtypes}')

print(f'df["Event Type"].unique():{df["Event Name"].unique()}')

ohe = OneHotEncoder()

feature_arry = ohe.fit_transform(df[feature_columns]).toarray()
print(f'feature_arry:{feature_arry}')
print(f'ohe.categories_:{ohe.categories_}')
# feature_labels = ohe.categories_
feature_labels = np.array(ohe.categories_).ravel()
print(f'feature_labels:{feature_labels}')

features = pd.DataFrame(feature_arry, columns=feature_labels)

print(f'features:{features}')
features.to_csv('/Users/yzhao/Downloads/test_1.csv', index=False)


df_new = pd.concat([df, features], axis=1)
print(f'df_new.columns:{df_new.columns}')

df_new.to_csv('/Users/yzhao/Downloads/test_2.csv', index=False)
