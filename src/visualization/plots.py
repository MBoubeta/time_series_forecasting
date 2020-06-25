import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt


def plot_loss(loss, val_loss, fig_name):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', color='k', alpha=.7, label='Train')
    plt.plot(epochs, val_loss, 'b', color='r', alpha=.7, label='Validation')
    plt.title('Train and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fig_name)


def plot_predictions(df_test, df_train=None, fig_name='./test/scaled_predictions.png'):

    # plot observed and predictions
    plt.figure(figsize=(10, 8))

    if df_train is None:
        plt.plot(df_test['ds'], df_test['y_test'], label='Observed')
        plt.plot(df_test['ds'], df_test['y_test_hat_arima'], label='ARIMA')
        plt.plot(df_test['ds'], df_test['y_test_hat_prophet'], label='Prophet')
        plt.plot(df_test['ds'], df_test['y_test_hat_lstm'], label='LSTM')
        plt.legend()

    else:
        # concatenate observed data
        df_test = df_test.rename(columns={"y_test": "y"})
        df_observed = pd.concat([df_train, df_test[['ds', 'y']]], axis=0, ignore_index=True)

        plt.plot(df_observed['ds'], df_observed['y'], label='Observed')
        plt.plot(df_test['ds'], df_test['y_test_hat_arima'], label='ARIMA')
        plt.plot(df_test['ds'], df_test['y_test_hat_prophet'], label='Prophet')
        plt.plot(df_test['ds'], df_test['y_test_hat_lstm'], label='LSTM')
        plt.legend()

    plt.savefig(fig_name)


def ts_chart(df, x='ds:T', y='y', title='Time series plot'):
    tschart = alt.Chart(df).mark_line(size=3, opacity=0.8).encode(
        x=x,
        y=y,
        tooltip=['ds:T', 'y']
    ).interactive().properties(width=900, height=450, title=title) \
        .configure_title(fontSize=20)
    # tschart.show()
    tschart.save('./test/ts_plot.html')


def plot_anomalies(df, title='Anomaly detection'):
    interval = alt.Chart(df).mark_area(interpolate="basis", color='#7FC97F').encode(
        x='ds:T',
        y='yhat_lower',
        y2='yhat_upper',
        tooltip=['ds:T', 'y', 'yhat_lower', 'yhat_upper']
    ).interactive().properties(
        title=title
    )

    y = alt.Chart(df[df.anomaly == 0]).mark_circle(size=15, opacity=0.9, color='Black').encode(
        x='ds:T',
        y='y',
        tooltip=['ds:T', 'y', 'yhat_lower', 'yhat_upper']
    ).interactive()

    anomalies = alt.Chart(df[df.anomaly != 0]).mark_circle(size=30, color='Red').encode(
        x='ds:T',
        y='y',
        tooltip=['ds:T', 'y', 'yhat_lower', 'yhat_upper'],
        size=alt.Size('importance', legend=None)
    ).interactive()

    anomaly_chart = alt.layer(interval, y, anomalies) \
        .properties(width=870, height=450) \
        .configure_title(fontSize=20)
    anomaly_chart.save('./test/anomaly_chart.html')
