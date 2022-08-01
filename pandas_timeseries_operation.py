import pandas as pd
import numpy as np
if __name__ == '__main__':
    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq='H')
    print(dt_rng)
