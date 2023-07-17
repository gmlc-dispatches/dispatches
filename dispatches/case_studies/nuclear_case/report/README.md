# Source Code for Generating Nuclear Case Study Results in Report

This folder contains the source code used to generate the results shown in the combined report for nuclear and renewable case studies. 

### `base_case_pcm_simulation.py`
This script performs a Prescient simulation of the base case i.e., nuclear generator without an electrolyzer. The script creates a new folder called `ne_without_pem_results` and saves the simulation results in that folder. Then, the LMP data at the bus `Attlee` is extracted from the file `bus_detail.csv`. For convenience, the extracted LMP data is included in the file `rts_gmlc_15_500.csv` file. This LMP data is used for the price-taker analysis.

### `traditional_tea.py`

This script performs the traditional techno-economic analysis. The function `run_exhaustive_enumeration` fixes the capacity of the electrolyzer and the selling price of hydrogen and computes the revenue from both electricity, and hydrogen markets and the annualized net present value. The script saves the results in a `json` file.

### `price_taker.py`

This script performs price-taker analysis. The function `run_exhaustive_enumeration` fixes the capacity of the electrolyzer and the selling price of hydrogen and computes the revenue from both electricity and hydrogen markets, and the annualized net present value. `market` argument to the function indicates which of the four price-taker variants to use for the analysis. To use the first variant (refer to the report for more details), specify `market="DA"`. For second, third, and fourth variants, specify `market="RT"`, `market="Max-DA-RT"`, and `market="DA-RT"`, respectively. The script saves the results in a `json` file.

### `market_surrogates.py`

Solves the conceptual design problem by embedding the surrogate models for market interactions. To run this script, `tensorflow` must be installed in the current environment. If it is not installed, the user can do so by running `pip install tensorflow` in a command window. The trained neural network surrogate models are included in the folder `nn_steady_state`. The function `run_exhaustive_enumeration` fixes the capacity of the electrolyzer and the selling price of hydrogen and computes the revenue from both electricity, and hydrogen markets and the annualized net present value, and  saves the results in a `json` file.
