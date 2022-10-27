import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randint(0, 2, size=(100, 4)), columns=list('ABCD'))

print(df)