import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)
import numpy as np

ny_iso_df = pd.read_csv("2020_Day-Ahead_Market_Generator_LBMP.csv")
generator = "INDIAN POINT___2"
nuclear_gen_data = ny_iso_df.loc[ny_iso_df['Gen Name'] == generator]
lmp = nuclear_gen_data["DAM Gen LBMP"]
lmp_np = lmp.to_numpy()
with open('ny_iso_results.npy', 'wb') as f:
    np.save(f,lmp_np)
