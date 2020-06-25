# libraries
import pickle
import numpy as np
import pandas as pd
import fbprophet as prop
from keras.models import load_model


def predict_arima(trained_model, predict_type, n_periods, df_train=None, df_test=None):

    assert predict_type in ('once', 'sequential')

    # load the trained model
    with open(trained_model, 'rb') as pkl:
        model = pickle.load(pkl)

    if predict_type is 'once':
        pred = model.predict(n_periods=n_periods)
    else:
        n_tot = df_test.shape[0]
        df_train_new = df_train.copy()
        pred = []

        for i in range(n_tot):

            model = model.fit(df_train_new)
            pred.append(model.predict(n_periods=1)[0])

            # update df_train
            df_train_new = pd.concat([df_train_new, df_test.iloc[[i]]])

        pred = np.array(pred)

    pred = pred.reshape((pred.shape[0], 1))
    return pred


def predict_prophet(trained_model, n_periods):
    # NOTE: prophet does not obtain sequential predictions. If train data is updated, a new model has to be fitted
    # load the trained model
    with open(trained_model, 'rb') as pkl:
        model = pickle.load(pkl)

    prophet_future = model.make_future_dataframe(periods=n_periods)
    pred = model.predict(prophet_future)

    return pred


def predict_lstm(trained_model, dat):
    # load model
    model = load_model(trained_model)

    # make predictions
    pred = model.predict(dat)
    return pred
