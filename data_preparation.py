import os
import pandas as pandas

MS_TO_MIN = 60_000
SHORT_LIMIT = 3
RECENT_ERA_START = 2020
DATASET_FILE = "top100_2010s_2020s.csv"

def get_base_dir():
	return os.path.dirname(os.path.abspath(__file__))

def get_output_dir(folder_name):
	path = os.path.join(get_base_dir(), folder_name)
	os.makedirs(path, exist_ok=True)
	return path

def get_dataset_path():
	return os.path.join(get_output_dir("dataset"), DATASET_FILE)

def load_data():
	dataframe = pandas.read_csv(get_dataset_path())

	dataframe["duration_min"] = dataframe["duration_ms"] / MS_TO_MIN
	dataframe["era"] = "2010-2019"
	dataframe.loc[dataframe["top_year"] >= RECENT_ERA_START, "era"] = "2020-2025"
	dataframe["is_short"] = dataframe["duration_min"] < SHORT_LIMIT

	return dataframe

def comma_format(value, decimals=2):
	return f"{value:.{decimals}f}".replace(".", ",")