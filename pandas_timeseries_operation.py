import pandas as pd
import numpy as np
if __name__ == '__main__':
    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq='H')
    print(dt_rng)

    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-03 23:00:00', freq='2H')
    print(dt_rng)

    dt_rng = pd.date_range('2015-03-02 00:00:00', '2015-03-02 3:00:00', freq='15min')
    print(dt_rng)
