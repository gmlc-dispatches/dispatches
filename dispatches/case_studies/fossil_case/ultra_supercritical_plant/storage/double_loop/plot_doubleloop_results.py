import os
from types import ModuleType
import pandas as pd
from importlib import resources
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
try:
    from importlib import resources  # Python 3.8+
except ImportError:
    import importlib_resources as resources  # Python 3.7
from dispatches.models.fossil_case.ultra_supercritical_plant import storage
from dispatches.models.fossil_case.ultra_supercritical_plant.storage import bidding_plugin_test_multiperiod_rankine

# result_dir = 'bidding_plugin_test_multiperiod_rankine3d2s_0420'



# comparison_result_dir = "no_plugin_result"
# result_dir = "double_loop_plugin_multiperiod_wind_battery"
# rts_dir = "/home/xgao1/DowlingLab/RTS-GMLC/RTS_Data/SourceData"

fossil_gen = "102_STEAM_3"
bus = 102

MAJOR_TICK_SIZE = "xx-large"
MINOR_TICK_SIZE = "xx-large"
LABEL_SIZE = "xx-large"
LEGEND_SIZE = "xx-large"
TITLE_SIZE = "xx-large"

# tracker_df = pd.read_csv(os.path.join(result_dir, "tracking_model_detail_new.csv"))
# tracker_df = pd.read_csv(os.path.join("tracking_results.csv"))
# tracker_df = tracker_df.loc[tracker_df["Horizon [hr]"]==0]
# tracker_df["Time Index"] = range(len(tracker_df))
# tracker_df

# fig, ax = plt.subplots(figsize=(10,5))
# tracker_df.plot(x="Time Index", y="Power Dispatch [MW]", ax=ax, label='RT Dispatches')
# tracker_df.plot(x="Time Index", y="Power Output [MW]", ax=ax, label='Power Output')

# ax.set_xlabel("Time [hr]", fontsize=LABEL_SIZE)
# ax.set_ylabel("Power Output [MW]", fontsize=LABEL_SIZE)
# ax.xaxis.set_minor_locator(MultipleLocator(4))
# ax.xaxis.set_major_locator(MultipleLocator(24))

# ax.tick_params(axis='both', which='major', labelsize=MAJOR_TICK_SIZE)
# ax.tick_params(axis='both', which='minor', labelsize=MINOR_TICK_SIZE)

# plt.rc('legend', fontsize = LEGEND_SIZE)
# plt.grid(False)
# plt.savefig('dispatch_results_3day_0413.png')

# ax.grid()
# ax.set_title(f"{fossil_gen} Plugin Power Output", fontsize=TITLE_SIZE)
with resources.path(storage, "tracking_results.csv") as data_file_path:
    assert data_file_path.is_file()
    tracking_model_df = pd.read_csv(str(data_file_path))
# with resources.path(bidding_plugin_test_multiperiod_rankine, "tracker_detail.csv") as data_file_path2:
#     assert data_file_path2.is_file()
#     tracking_model_df2 = pd.read_csv(str(data_file_path2))

# tracking_model_df = pd.read_csv(os.path.join(result_dir, "tracking_model_detail_new.csv"))
tracking_model_df = tracking_model_df.loc[tracking_model_df["Horizon [hr]"]==0]
# tracking_model_df2 = tracking_model_df2.loc[tracking_model_df2["Horizon [hr]"]==0]
tracking_model_df["Time Index"] = range(len(tracking_model_df))
# tracking_model_df
tracking_model_df["Storage Tank Level [%]"] = tracking_model_df["Hot Tank Level [MT]"]/67392.92
tracking_model_df["Boiler Duty [MWth]"] = tracking_model_df["Plant Heat Duty [MWth]"]

fig, ax = plt.subplots(figsize=(12, 6))
ax2 = ax.twinx()
cols = ["Plant Power [MW]", "Storage Power [MW]", "Time Index"]
ylabels = ["Plant Power [MW]", "Storage Power [MW]"]
tracking_model_df[cols].plot(x="Time Index", kind='bar', width=1, stacked=True, ax=ax, label=ylabels, ylim=(0, 600))
# tracking_model_df[cols].plot.area(x="Time Index", ax=ax)
# tracking_model_df.plot(x="Time Index", y="Power Dispatch [MW]", drawstyle="steps-mid",ax=ax, label='RT Dispatches [MWe]', color="black", ylim=(0, 600))
# plt.savefig('tracking_dispatch_results_3day.png')
# plt.show()

tracking_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, label='Storage Level [%]', color="red", ylim=(0, 600))
tracking_model_df.plot(x="Time Index", y="Boiler Duty [MWth]", ax=ax, label='Boiler Duty [$MW_{th}$]', color="pink", ylim=(0, 600))
# legend1 = pyplot.legend(plot_lines[0], ["algo1", "algo2", "algo3"], loc=1)
# tracking_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, color="red")
# tracking_model_df.plot(x="Time Index", y="Power Prices [$/MWh]", ax=ax2, label='LMP [$/MWh]', color="gold", ylim=(0, 40))
# legend1 = plt.legend(fontsize=LEGEND_SIZE, ncol=2, loc = "upper left", bbox_to_anchor=(0.7,1.2))
# tracker_df.legend(fontsize=LEGEND_SIZE, ncol=2, loc = "upper left", bbox_to_anchor=(0.7,1.2))
ax.set_xlabel("Time [hr]", fontsize=LABEL_SIZE)
# ax.set_ylabel("Power [MW]", fontsize=LABEL_SIZE)
ax2.set_ylabel("LMP [$/MWh]", fontsize=LABEL_SIZE)
ax.xaxis.set_minor_locator(MultipleLocator(4))
ax.xaxis.set_major_locator(MultipleLocator(24))
ax2.xaxis.set_minor_locator(MultipleLocator(4))
ax2.xaxis.set_major_locator(MultipleLocator(24))
ax.legend(fontsize=LEGEND_SIZE, ncol=2, loc = "upper left", bbox_to_anchor=(0.26,1.3))

ax.tick_params(axis='both', which='major', labelsize=MAJOR_TICK_SIZE)
ax.tick_params(axis='both', which='minor', labelsize=MINOR_TICK_SIZE)
ax2.tick_params(axis='both', which='minor', labelsize=MINOR_TICK_SIZE)
ax2.tick_params(axis='both', which='major', labelsize=MAJOR_TICK_SIZE)

# ax.grid()
# plt.ylim([0, 500])
# plt.grid(False)
plt.legend(fontsize=LEGEND_SIZE, ncol=2, loc = "upper left", bbox_to_anchor=(-0.01,1.2))
plt.savefig('doubleloop_standard_0422_3d.png')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax2 = ax.twinx()
# cols = ["Plant Power [MW]", "Storage Power [MW]", "Time Index"]
# tracking_model_df[cols].plot(x="Time Index", kind='bar', width=1, stacked=True, ax=ax)
# tracking_model_df[cols].plot.area(x="Time Index", ax=ax)
# tracker_df.plot(x="Time Index", y="Power Dispatch [MW]", drawstyle="steps-mid",ax=ax, label='RT Dispatches', color="black")
# plt.savefig('tracking_dispatch_results_3day.png')
# plt.show()
# tracking_model_df.plot(x="Time Index", y="Power Prices [$/MWh]", ax=ax, label='LMP [$/MWh]', color="green", ylim=(0,40))
# tracking_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, label='Tank Level [%]', color="red")
tracking_model_df.plot(x="Time Index", y="Hot Tank Level [MT]", drawstyle="steps-mid",ax=ax2, label='Tank Level [MT]', color="red", ylim=(0,3e6))
# 
# tracking_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, color="red")
# tracking_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, color="red")

ax.set_xlabel("Time [hr]", fontsize=LABEL_SIZE)
# ax.set_ylabel("LMP [$/MWh]", fontsize=LABEL_SIZE)
ax2.set_ylabel("Salt Level [MT]", fontsize=LABEL_SIZE)
ax.xaxis.set_minor_locator(MultipleLocator(4))
ax.xaxis.set_major_locator(MultipleLocator(24))

ax.tick_params(axis='both', which='major', labelsize=MAJOR_TICK_SIZE)
ax.tick_params(axis='both', which='minor', labelsize=MINOR_TICK_SIZE)

# ax.grid()
# plt.legend(fontsize=LEGEND_SIZE, ncol=2, loc = "upper center", bbox_to_anchor=(0.5,1.35))
# plt.ylim([0, 500])
# plt.grid(False)
plt.savefig('LMP_SaltLevel_0422_3d.png')
plt.show()
# ****************************************************

# ax.set_title(f"{fossil_gen} Plugin Tracking Power States", fontsize=TITLE_SIZE)

# bidding_model_df = pd.read_csv(os.path.join(result_dir, "bidding_model_detail_3day.csv"))
# bidding_model_df = bidding_model_df.loc[bidding_model_df["Horizon [hr]"]<24]
# bidding_model_df["Time Index"] = range(len(bidding_model_df))
# #bidding_model_df

# fig, ax = plt.subplots(figsize=(10, 5))
# cols = ["Plant Power [MW]", "Storage Power [MW]", "Time Index"]
# bidding_model_df[cols].plot(x="Time Index", kind='bar', width=1, stacked=True, ax=ax)
# bidding_model_df[cols].plot.area(x="Time Index", ax=ax)
# # tracker_df.plot(x="Time Index", y="Power Dispatch [MW]", ax=ax, label='RT Dispatches', color="black")
# # bidding_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, color="red")
# bidding_model_df.plot(x="Time Index", y="Total Power Output [MW]", drawstyle="steps-mid", ax=ax, color="black")

# ax.set_xlabel("Time [hr]", fontsize=LABEL_SIZE)
# ax.set_ylabel("Power [MW]", fontsize=LABEL_SIZE)
# ax.xaxis.set_minor_locator(MultipleLocator(4))
# ax.xaxis.set_major_locator(MultipleLocator(24))

# ax.tick_params(axis='both', which='major', labelsize=MAJOR_TICK_SIZE)
# ax.tick_params(axis='both', which='minor', labelsize=MINOR_TICK_SIZE)

# plt.rc('legend', fontsize = LEGEND_SIZE)

# ax.grid()
# ax.set_title(f"{fossil_gen} Plugin Bidding Power States", fontsize=TITLE_SIZE)

fig, ax = plt.subplots(figsize=(10, 5))
ax2 = ax.twinx()
# cols = ["Plant Power [MW]", "Storage Power [MW]", "Time Index"]
# tracking_model_df[cols].plot(x="Time Index", kind='bar', width=1, stacked=True, ax=ax)
# tracking_model_df[cols].plot.area(x="Time Index", ax=ax)
# tracker_df.plot(x="Time Index", y="Power Dispatch [MW]", drawstyle="steps-mid",ax=ax, label='RT Dispatches', color="black")
# plt.savefig('tracking_dispatch_results_3day.png')
# plt.show()
# tracking_model_df.plot(x="Time Index", y="Power Prices [$/MWh]", ax=ax, label='LMP [$/MWh]', color="green", ylim=(0,40))
tracking_model_df.plot(x="Time Index", y="HXC Duty", drawstyle="steps-mid", ax=ax2, label='Charge Heat Exchanger [$MW_{th}$]', color="red", ylim=(0,200))
tracking_model_df.plot(x="Time Index", y="HXD Duty", drawstyle="steps-mid",ax=ax2, label='Discharge Heat Exchanger [$MW_{th}$]', color="blue", ylim=(0,200))

ax.legend(loc = "upper left", bbox_to_anchor=(0.01,1))

# tracking_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, color="red")
# tracking_model_df.plot(x="Time Index", y="Storage Tank Level [%]", drawstyle="steps-mid", ax=ax, color="red")

ax.set_xlabel("Time [hr]", fontsize=LABEL_SIZE)
# ax.set_ylabel("LMP [$/MWh]", fontsize=LABEL_SIZE)
ax2.set_ylabel("Heat Duty [$MW_{th}$]", fontsize=LABEL_SIZE)
ax.xaxis.set_minor_locator(MultipleLocator(4))
ax.xaxis.set_major_locator(MultipleLocator(24))

ax.tick_params(axis='both', which='major', labelsize=MAJOR_TICK_SIZE)
ax.tick_params(axis='both', which='minor', labelsize=MINOR_TICK_SIZE)

# ax.grid()
# plt.legend(fontsize=LEGEND_SIZE, ncol=2, loc = "upper center", bbox_to_anchor=(0.5,1.35))
# plt.ylim([0, 500])
# plt.grid(False)
plt.savefig('hx_duties_0422_3d.png')
plt.show()
