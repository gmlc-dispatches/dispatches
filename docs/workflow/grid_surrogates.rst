Surrogate Models for Grid Outcomes
==================================


1. The Steady-State Co-Optimization with Market Interactions
-------------------------------------------------------------

We developed two surrogate model architectures to map market input parameters to market outputs. Each surrogate takes 8 inputs from the production cost model (PCM) Prescient
outlined in table below. The data for training the surrogates is from the Prescient sensitivity analysis.



=================  ==============================================  ===================
Input              Descriptions                                     Units
=================  ==============================================  ===================
x\ :sub:`1`\       Maximum Designed Capacity (P\ :sub:`max`\)      MW
x\ :sub:`2`\       Minimum Operating Multiplier                    --
x\ :sub:`3`\       Ramp Rate Multiplier                            --
x\ :sub:`4`\       Minimum Up Time                                 hr
x\ :sub:`5`\       Minimum Down Multiplier                         --
x\ :sub:`6`\       Marginal Cost                                   $/MWh
x\ :sub:`7`\       No Load Cost                                    $/hr@P\ :sub:`max`\
x\ :sub:`8`\       Representative Startup Cost                     $/MW capacity
=================  ==============================================  ===================


=================  ==============================================  ===================
Output              Descriptions                                     Units
=================  ==============================================  ===================
y\ :sub:`1`\       Annual Revenue                                  MM $
y\ :sub:`2`\       Annual Number of Startups                       #
y\ :sub:`z`\       Annual Hours Dispatched in zone z               hr
=================  ==============================================  ===================

Market revenue y\ :sub:`1`\   is a surrogate function of the bid parameters, **x**, which correspond to the data which
each individual resource communicates to the wholesale electricity market. y\ :sub:`2`\  approximates the number of
startups of the generator during the simulation time periods. y\ :sub:`z`\  is the surrogate for frequency of each scenario,
we use eleven total zones to represent generator power output scaled by the nameplate capacity (maximum power output).
The zones consist of an ’off’ state and ten power outputs between the minimum and maximum output of the generator, i.e., 0-10%, 10-20%, ..., 90-100%.


2. ALAMO Surrogate Models
---------------------------------
We use ALAMO (version 2021.12.28) (https://idaes-pse.readthedocs.io/en/1.4.4/apps/alamopy.html) to train algebraic
surrogates which consists of a linear combination of nonlinear basis functions x\ :sub:`j`\  and regressed coefficients
for coefficient :math:`\beta`\ \ :sub:`j`\  for index j in set B

.. math:: y_alamo = \sum_{j \in \beta} \beta_j X_j(x)

For training, ALAMO considers monomial and binomial basis functions with up to 15 total terms with power values of 1, 2,
and 3. We use Bayesian Information Criteria (BIC) implemented in ALAMO to select the best algebraic surrogate using
enumeration mode. In total, we train a total of fourteen surrogate models using the ALAMO version accessible through the
**IDAES-PSE** interface: revenue (one), number of startups (one), and surrogates for each zone (eleven).

Three ALAMO surrogate models are trained in 'train_nstartups_idaes.py', 'train_revenue_idaes.py' and 'train_zones_idaes.py'.
The input training data can be read in or simulated using available Python packages and 1/3 of the training data are
withheld for testing the model. The data are normalized before fed to the trainer. There are no other arguments
needed to specify the training. ALAMO solves ordinary least squares regression problems and generates the output results
in the json files. (The ALAMO training options are default set in 'train_nstartups/revenue/zones.py') There will be three output json
files. The 'alamo_nstartups/revenue/zones.json' stores the coefficients of the monomial and binomial basis functions.
The 'alamo_parameters_nstartups/revenue/zones.json' saves scaling and training bounds for the input data.
The 'alamo_nstartups/revenue/zones_accuracy.json' has the computed R\ :sup:`2`\  matrices.

3. Neural Network (NN) Surrogate Models
--------------------------------------------
Feed-forward neural network (NN) surrogate models are trained.

.. math:: x = z_0

.. math:: z_k = \sigma(W_k z_{k-1} + b_k), k\in \{1,2,...,K-1\}

.. math:: y_{nn} = W_k z_{k-1} + b_k

We use the 'MLPRegressor' package (Keras version v2.8.0, Scikit Learn version v0.24.2) with default settings to train three
2-layer neural networks.The revenue and startup surrogates contain two hidden layers with 100 nodes in the first hidden
layer and 50 nodes in the second (for the annual zone output surrogate, 100 nodes in both layers).

Three NN surrogate models are trained in 'train_nstartups.py', 'train_revenue.py' and 'train_zones.py'. The input training data
can be read in or simulated using available python packages and 1/3 of the training data are split for testing the
model. The data are normalized before fed to the trainer. There are no other arguments needed to specify the
training. There are two output json files and one pickle file that save the results. The 'scikit_nstartups/revenue/zones.pkl' stores the
coefficients of the neural networks. 'The scikit_parameters_nstartups/revenue/zones.json' saves scaling and training bounds
for the input data. The 'scikit_nstartups/revenue/zones_accuracy.json' has the computed R\ :sup:`2`\  matrices.

The accuracy of the scikit NN surrogate models can be visualized by 'plot_scikit_nstartups/revenue/zones.py'.

A Jupyter Notebook demonstration can be found in the following link:
https://github.com/jalving/dispatches/blob/prescient_verify/dispatches/workflow/surrogate_design/rankine_cycle_case/grid_surrogate_design.ipynb

4. Optimization with Surrogate Models
---------------------------------------
We can implement the steady-state co-optimization with market interactions in part 1 using 'run_surrogate_alamo.py' and
'run_surrogate_nn.py'. The scripts formulate the optimization using Pyomo and use Python packages to add the surrogate
model coefficients and input data bounds from the json and pickle files. Optionally, some surrogate inputs may be fixed
(removed as optimization degrees of freedom) in the scripts. The optimization solution is stored in
'conceptual_design_solution_alamo/nn.json's which can be read by the Prescient for further verification.




