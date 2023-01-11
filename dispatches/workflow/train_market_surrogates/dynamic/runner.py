#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
import os

file_name = 'test_automation.txt'

with open(file_name, "w") as f:
    f.write(
        "#!/bin/bash\n"
        + "#$ -M xchen24@nd.edu\n"
        + "#$ -m ae\n"
        + "#$ -q long\n"
        + f"#$ -N test_automation\n"
        + "conda activate regen\n"
        + "python dy_surrogate_automation.py")

os.system(f"qsub {file_name}")
print(f'{file_name} has been submitted.')