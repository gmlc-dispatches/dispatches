import numpy as np
import copy
from PySAM.ResourceTools import SRW_to_wind_data


with open('/Users/dguittet/Projects/Dispatches/idaes-pse/idaes/apps/multiperiod/examples/rts_results_all_prices.npy', 'rb') as f:
    dispatch = np.load(f)
    price = np.load(f)

prices_used = copy.copy(price)
prices_used[prices_used > 200] = 200
weekly_prices = prices_used.reshape(52, 168)
n_time_points=7*24

# simple financial assumptions
i = 0.05    # discount rate
N = 30      # years
PA = ((1+i)**N - 1)/(i*(1+i)**N)    # present value / annuity = 1 / CRF

# wind data
wind_data = SRW_to_wind_data('44.21_-101.94_windtoolkit_2012_60min_80m.srw')
wind_speeds = [wind_data['data'][i][2] for i in range(8760)]

wind_resource = {t:
                     {'wind_resource_config': {
                         'resource_probability_density': {
                             0.0: ((wind_speeds[t] * 1.1, 180, 0.5),
                                   (wind_speeds[t] * 1.5, 180, 0.5))}}} for t in range(n_time_points)}
# wind_resource = {t: {'wind_resource_config': None} for t in range(n_time_points)}

