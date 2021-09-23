#This script was used to populate the scenario directory as opposed to using the existing symlinks (which causing issues on windows)
import os
import shutil

#Assumine we start at `deterministic_scenarios`, list the folders representing each day
dirs = []
for path in os.listdir():
	if os.path.isdir(path):
		dirs.append(path)

#this file is in `deterministic_scenarios`. we will copy it into each subfolder.
scenario_structure = os.path.abspath("./")+"/ScenarioStructure.dat" 
for dir in dirs:
	#we also copy the forecast file to the actuals file for each day (folder)
	forecasts_file = os.path.abspath(dir)+"/Scenario_forecasts.dat"
	actuals_file = os.path.abspath(dir)+"/Scenario_actuals.dat"
	structure_file = os.path.abspath(dir)+"/ScenarioStructure.dat"
	shutil.copyfile(forecasts_file,actuals_file)
	shutil.copyfile(scenario_structure,structure_file)
	