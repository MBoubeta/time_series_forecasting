# libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.constants import *
from src.data.load_input_data import load_input_data
from src.feature_engineering.features import univariate_data, split_train_test
from src.models.anomaly_detection import fit_prophet_model, detect_anomalies
from src.visualization.plots import ts_chart, plot_anomalies


def anomaly_detection(file_name: str, method: str):
    # load input data
    df = load_input_data(file_name)

    # scale data
    scale = MinMaxScaler(feature_range=(-1, 1))
    df_scale = scale.fit_transform(df)

    ##  split into train and test sets
    # train size: percentage of observations to consider in the train set
    train_percentage = 0.67
    train_split = int(len(df) * train_percentage)

    ## ------------------------------------------
    ## INPUTS
    ## ------------------------------------------
    # input for statistics models
    df_scale_train, df_scale_val = split_train_test(df=df_scale, train_split=train_split)

    # input reshape for LSTMs
    x_train, y_train = univariate_data(dataset=df_scale, start_index=0, end_index=train_split,
                                       size_window=size_window, target_size=target_size)
    x_val, y_val = univariate_data(dataset=df_scale, start_index=train_split - size_window, end_index=None,
                                   size_window=size_window, target_size=target_size)

    if method == "Prophet":
        ts_chart(df=df_scale_train, x='ds:T', y='y', title='Time series plot')

        # fit a Prophet model
        prophet_pred = fit_prophet_model(df_scale_train)
        prophet_pred = detect_anomalies(prophet_pred)
        plot_anomalies(prophet_pred, title='Anomaly detection')
    elif method == "LSTM":
        asasd


















