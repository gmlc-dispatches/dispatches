### Files contained:

- ARMA_train.xml: 	RAVEN input file that trains ARMA model based on given data
- Price_20xx.csv:	  csv files with **synthetic** training data (2018-2021)
- ARMA_pointer_2018_2021_to2045: 	csv file pointing to training data (2018-2021, interpolate through 2045)

## Training the ARMA Model:

Once `raven_framework` is properly installed, run:

  ```bash
    raven_framework ARMA_train.xml
  ```
in this directory from an open terminal. The arma file will be found in `output/arma.pk` with some additional training metadata.

## Note

Please keep in mind this ARMA model is intended for demonstration use only, no meaningful conclusions should be derived from simulation results.

Check the RAVEN github page for updates on ARMA models; if there are errors, be sure to re-train the ARMA model by re-running the ARMA_train.xml script with RAVEN.

For more information on how to run an XML input file within RAVEN for ARMA training, see https://github.com/idaholab/raven/tree/devel/doc/workshop/ARMA.


