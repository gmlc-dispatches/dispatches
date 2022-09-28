### Files contained:

- output/arma.pk: 	binary file containing trained ARMA model
- ARMA_train.xml: 	RAVEN input file that trains ARMA model based on given data
- Price_20xx.csv:	  csv files with **synthetic** training data (2018-2021)
- ARMA_pointer_2018_2021_to2045: 	csv file pointing to training data (2018-2021, interpolate through 2045)

Please keep in mind this ARMA model is intended for demonstration use only, no meaningful conclusions should be derived from simulation results.

Check the RAVEN github page for updates on ARMA models; if there are errors, be sure to re-train the ARMA model by re-running the ARMA_train.xml script with RAVEN.

For more information on how to run an XML input file within RAVEN for ARMA training, see https://github.com/idaholab/raven/tree/devel/doc/workshop/ARMA.


