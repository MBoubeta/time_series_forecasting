# libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.constants import *
from src.data.load_input_data import load_input_data
from src.feature_engineering.features import univariate_data, split_train_test
from src.models.predict import predict_arima, predict_prophet, predict_lstm
from src.visualization.plots import plot_predictions


def predict_process(file_name: str):
    # input data
    df = load_input_data(file_name)

    # scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scale = scaler.fit_transform(df)

    ##  split into train and test sets
    # train size: percentage of observations to consider in the train set
    train_percentage = 0.67
    train_split = int(len(df) * train_percentage)

    # input for statistics models
    df_scale_train, df_scale_val = split_train_test(df=df_scale, train_split=train_split)

    # input reshape for LSTMs
    x_test, y_test = univariate_data(dataset=df_scale, start_index=train_split - size_window, end_index=None,
                                     size_window=size_window, target_size=target_size)

    # get predictions
    n_periods = df_scale_val.shape[0]
    y_test_hat_arima = predict_arima(trained_model=arima_name, predict_type='sequential', n_periods=n_periods,
                                     df_train=df_scale_train["y"], df_test=df_scale_val["y"])
    y_test_hat_prophet = predict_prophet(trained_model=prophet_name, n_periods=n_periods)
    y_test_hat_lstm = predict_lstm(trained_model=lstm_name, dat=x_test)

    y_test_hat_prophet = y_test_hat_prophet.iloc[train_split:, ]

    df_test = pd.DataFrame({'ds': list(np.array(y_test_hat_prophet["ds"])),
                            'y_test': list(y_test.flatten()),
                            'y_test_hat_arima': list(y_test_hat_arima.flatten()),
                            'y_test_hat_prophet': list(np.array(y_test_hat_prophet["yhat"])),
                            'y_test_hat_lstm': list(y_test_hat_lstm.flatten())})
    df_test_cols = df_test.columns[df_test.columns != 'ds']

    # plot scaled predictions: df_train=None or df_train=df_scale_train
    plot_predictions(df_test=df_test, df_train=df_scale_train, fig_name='./test/scaled_predictions.png')

    # invert predictions
    df_test = pd.concat([
        df_test.loc[:, ['ds']],
        pd.DataFrame(scaler.inverse_transform(df_test.loc[:, ~df_test.columns.isin(['ds'])]),
                     columns=df_test_cols)],
        axis=1)
    df_scale_train.loc[:, ['y']] = scaler.inverse_transform(df_scale_train.loc[:, ['y']])

    # plot scaled predictions: df_train=None or df_train=df_scale_train
    plot_predictions(df_test=df_test, df_train=None, fig_name='./test/predictions.png')

    return df_test
