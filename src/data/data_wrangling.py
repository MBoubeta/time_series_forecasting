# libraries
import pandas as pd


def data_prepare(df: pd.DataFrame):
    """
    Prepare the data to be used in the models
    :param df: data frame to be transformed
    :return: the transformed data frame
    """

    dat = df.values
    dat = dat.astype('float')
    return dat
