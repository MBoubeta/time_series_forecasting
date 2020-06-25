import math
from sklearn.metrics import mean_squared_error


def RMSE(observed, predicted):
    score = math.sqrt(mean_squared_error(observed, predicted))
    return score
