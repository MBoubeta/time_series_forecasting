# libraries
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union


def split_train_test(df: pd.DataFrame, train_split: int):
    """
    Split a data frame into train and test sets
    :param df: a data frame to split
    :param train_split: split index
    :return: train and test data frames
    """

    if type(df) is np.ndarray:
        df = pd.DataFrame(data=df, columns=['y'])

    if df.shape[1] == 1:
        df['ds'] = pd.date_range(end=datetime.today(), periods=len(df))

    # order columns
    df = df[["ds", "y"]]

    # split into train and test sets
    train, test = df.iloc[0:train_split, :], df.iloc[train_split:len(df), :]
    return train, test


def univariate_data(dataset: pd.DataFrame, start_index: int, end_index: Union[int, None],
                    size_window: int, target_size: int = 0):
    """
    Split a dataset into features and labels.
    :param dataset: dataset to split.
    :param start_index: start index.
    :param end_index: end index.
    :param size_window: size of the past window of information.
    :param target_size: label that needs to be predicted.
    :return: two np.arrays with features and labels datasets.
    """

    data = []
    labels = []

    start_index = start_index + size_window
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - size_window, i)

        # reshape data from (size_window,) to (size_window, 1)
        data.append(np.reshape(dataset[indices], (size_window, 1)))
        labels.append(dataset[i + target_size])

    data = np.array(data)
    labels = np.array(labels)
    return data, labels
