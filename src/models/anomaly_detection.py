# libraries
import numpy as np
import pandas as pd
import altair as alt
from fbprophet import Prophet
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential


def fit_prophet_model(data, interval_width=0.99, changepoint_range=0.8):
    m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=interval_width,
                changepoint_range=changepoint_range)
    m = m.fit(data)

    forecast = m.predict(data)
    forecast['y'] = data['y'].reset_index(drop=True)
    print('Prophet plot')
    fig1 = m.plot(forecast)
    return forecast


def detect_anomalies(forecast):
    forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'y']].copy()

    # calculate anomalies
    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['y'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['y'] < forecasted['yhat_lower'], 'anomaly'] = -1

    # calculate anomaly importances
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] == 1, 'importance'] = \
        (forecasted['y'] - forecasted['yhat_upper']) / forecast['y']
    forecasted.loc[forecasted['anomaly'] == -1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['y']) / forecast['y']

    return forecasted


def autoencoder_model(x):
    inputs = Input(shape=(S.shape[1], X.shape[2]))
    L1 = LSRM(16, activation='relu', return_sequences=True, kernel_regularized=regularizers.l2(0.00))(inputs)






class LSTM_Autoencoder:
    def __init__(self, optimizer='adam', loss='mse'):
        self.optimizer = optimizer
        self.loss = loss
        self.n_features = 1

    def build_model(self):
        timesteps = self.timesteps
        n_features = self.n_features
        model = Sequential()

        # Encoder
        model.add(LSTM(timesteps, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(LSTM(1, activation='relu'))
        model.add(RepeatVector(timesteps))

        # Decoder
        model.add(LSTM(timesteps, activation='relu', return_sequences=True))
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))

        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.summary()
        self.model = model

    def fit(self, X, epochs=3, batch_size=32):
        self.timesteps = X.shape[1]
        self.build_model()

        input_X = np.expand_dims(X, axis=2)
        self.model.fit(input_X, input_X, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        input_X = np.expand_dims(X, axis=2)
        output_X = self.model.predict(input_X)
        reconstruction = np.squeeze(output_X)
        return np.linalg.norm(X - reconstruction, axis=-1)

    def plot(self, scores, timeseries, threshold=0.95):
        sorted_scores = sorted(scores)
        threshold_score = sorted_scores[round(len(scores) * threshold)]

        plt.title("Reconstruction Error")
        plt.plot(scores)
        plt.plot([threshold_score] * len(scores), c='r')
        plt.show()

        anomalous = np.where(scores > threshold_score)
        normal = np.where(scores <= threshold_score)

        plt.title("Anomalies")
        plt.scatter(normal, timeseries[normal][:, -1], s=3)
        plt.scatter(anomalous, timeseries[anomalous][:, -1], s=5, c='r')
        plt.show()


lstm_autoencoder = LSTM_Autoencoder(optimizer='adam', loss='mse')
lstm_autoencoder.fit(normal_timeseries, epochs=3, batch_size=32)
scores = lstm_autoencoder.predict(test_timeseries)
lstm_autoencoder.plot(scores, test_timeseries, threshold=0.95)


