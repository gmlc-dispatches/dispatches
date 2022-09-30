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