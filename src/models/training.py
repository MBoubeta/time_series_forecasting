# libraries
import numpy as np
import pmdarima as pm
import fbprophet as prop
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def fit_arima(df, start_p=1, d=None, start_q=1, max_p=5, max_d=2, max_q=5, start_P=0, D=None, start_Q=0,
              max_P=2, max_D=1, max_Q=2, m=1, seasonal=True, test='adf', stepwise=True, trace=True,
              error_action='ignore', suppress_warnings=True, model_name='./models/arima.pkl'):

    model = pm.auto_arima(df, start_p=start_p, d=d, start_q=start_q, max_p=max_p, max_d=max_d, max_q=max_q,
                          start_P=start_P, D=D, start_Q=start_Q, max_P=max_P, max_D=max_D, max_Q=max_Q,
                          m=m, seasonal=seasonal, test=test, stepwise=stepwise, trace=trace,
                          error_action=error_action, suppress_warnings=suppress_warnings)

    # save the trained model
    with open(model_name, 'wb') as pkl:
        pickle.dump(model, pkl)

    return model


def fit_prophet(df, model_name='./models/prophet.pkl'):
    model = prop.Prophet()
    model.fit(df)

    # save the trained model
    with open(model_name, 'wb') as pkl:
        pickle.dump(model, pkl)

    return model


def fit_lstm(size_window: int, x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array,
             model_name: str = './models/lstm.h5'):
    """
    Create and fit the LSTM network
    :param size_window: lag to consider
    :param x_train: train auxiliary information
    :param y_train: train response
    :param x_val: test auxiliary information
    :param y_val: test response
    :param model_name: file name to save the model
    :return: a fitted LSTM model
    """

    model = Sequential()
    model.add(LSTM(4, input_shape=(size_window, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_data=(x_val, y_val))

    # save the trained model
    model.save(model_name)
    return model


