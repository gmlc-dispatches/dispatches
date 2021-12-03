import numpy as np
import copy
from PySAM.ResourceTools import SRW_to_wind_data

# constants
h2_mols_per_kg = 500

# costs
wind_cap_cost = 1550
wind_op_cost = 43
batt_cap_cost = 1000 + 500 * 4
pem_cap_cost = 1630
pem_op_cost = 47.9
pem_var_cost = 1.3/1000
h2_price_per_kg = 5.9

# sizes
fixed_wind_mw = 200
wind_ub_mw = 500
fixed_batt_mw = 100
fixed_pem_mw = 20

# operation parameters
pem_bar = 8
battery_ramp_rate = 300

# prices
with open('/Users/dguittet/Projects/Dispatches/idaes-pse/idaes/apps/multiperiod/examples/rts_results_all_prices.npy', 'rb') as f:
    dispatch = np.load(f)
    price = np.load(f)

prices_used = copy.copy(price)
prices_used[prices_used > 200] = 200
weekly_prices = prices_used.reshape(52, 168)
n_time_points = 7*24
# n_time_points = 24
h2_contract = False

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
                             0.0: ((min(wind_speeds[t], 24), 180, 0.5),
                                   (min(wind_speeds[t], 24), 180, 0.5))}}} for t in range(n_time_points)}
# wind_resource = {t: {'wind_resource_config': None} for t in range(n_time_points)}
x = 5

