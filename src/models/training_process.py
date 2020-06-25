# libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.constants import *
from src.data.load_input_data import load_input_data
from src.feature_engineering.features import univariate_data, split_train_test
from src.models.training import fit_arima, fit_prophet, fit_lstm
from src.models.predict import predict_arima, predict_prophet
from src.utils.metrics import RMSE
from src.visualization.plots import *


def training_process(file_name: str):
    """
    Function training_process trains the considered time series models
    :param file_name: file name of the .csv that contains the data
    :return: the saved models and a DataFrame with the evaluation metrics computed
    """

    # set seed for reproducibility
    np.random.seed(7)
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

    ## ------------------------------------------
    ## TRAIN MODELS
    ## ------------------------------------------
    # train statistics models
    arima_mod = fit_arima(df=df_scale_train["y"], model_name=arima_name)
    prophet_mod = fit_prophet(df=df_scale_train, model_name=prophet_name)

    # train the recurrent neural networks models
    lstm_mod = fit_lstm(size_window, x_train, y_train, x_val, y_val, model_name=lstm_name)

    ## ------------------------------------------
    ## EVALUATE THE MODELS
    ## ------------------------------------------
    # evaluate the models
    n_periods = df_scale_val.shape[0]
    arima_pred = predict_arima(trained_model=arima_name, predict_type='sequential', n_periods=n_periods,
                               df_train=df_scale_train["y"], df_test=df_scale_val["y"])
    arima_eval = RMSE(observed=df_scale_val["y"], predicted=arima_pred)

    prophet_pred = predict_prophet(trained_model=prophet_name, n_periods=n_periods)
    prophet_eval = RMSE(observed=df_scale_val["y"], predicted=prophet_pred.loc[train_split:, ['yhat']])

    plot_loss(loss=lstm_mod.history.history['loss'], val_loss=lstm_mod.history.history['val_loss'],
              fig_name=fig_name_loss)
    lstm_eval = np.sqrt(lstm_mod.evaluate(x_val, y_val))

    models_eval = pd.DataFrame({'model': ['ARIMA', 'Prophet', 'LSTM'],
                                'RMSE': [arima_eval, prophet_eval, lstm_eval]})
    return models_eval
