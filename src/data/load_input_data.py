# libraries
import pandas as pd
from config.constants import *
from src.data.data_wrangling import data_prepare


def load_input_data(file_name):
    # input data
    df = pd.read_csv(file_name, usecols=[1], engine='python')

    # data wrangling
    df = data_prepare(df)

    return df

