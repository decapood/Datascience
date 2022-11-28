import pandas as pd
import numpy as np

from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def MAPE(actual, prediction):
    actual, prediction = np.array(actual), np.array(prediction)
    return np.mean(np.abs((actual - prediction)/actual))* 100

def predict_ARIMA(name,training_set,test_set, key,type):
    model = auto_arima(y = training_set[f"{key}"], seasonal=True, stepwise=False)
    predicitons_arima = pd.Series(model.predict(n_periods= len(test_set), X = test_set,m = 7)).rename("SARIMA")
    predicitons_arima.index = test_set.index
    plt.plot(training_set[f"{key}"])
    plt.plot(predicitons_arima)
    plt.savefig(f"Results/SARIMA/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_ARIMA_Predictions.png")
    plt.clf()
    # training_set[f"{key}"].plot(figsize=(9,6),legend=True)
    # test_set[f"{key}"].plot(legend=True)
    # predicitons_arima.plot(legend=True)
    # print(round(mean_absolute_error(test_set.revenue,predicitons_arima),0))
    # print(round(np.sqrt(mean_squared_error(test_set.revenue,predicitons_arima))))
    predicitons_arima.to_csv(f"Results/SARIMA/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_ARIMA_Predictions.csv")
    return predicitons_arima #, MAPE(test_set['quantity'],predicitons_arima)

def predict_HW(name,training_set,test_set,key,type):
    if type =="Monthly":
        model = ExponentialSmoothing(endog = training_set[f"{key}"]).fit()
    else:
        model = ExponentialSmoothing(endog = training_set[f"{key}"],trend='add',seasonal='add',seasonal_periods = 7).fit()
    predictions = model.forecast(steps= len(test_set)).rename('prediction')
    plt.plot(training_set[f"{key}"])
    plt.plot(predictions)
    plt.savefig(f"Results/Holt_Winters/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_HW_Predictions.png")
    plt.clf()
    predictions.to_csv(f"Results/Holt_Winters/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_HW_Predictions.csv")
    return predictions #, MAPE(test_set.revenue,predictions)

def predict_ARIMA_MAPE(name,training_set,test_set, key,type):
    model = auto_arima(y = training_set[f"{key}"])
    predictions = pd.DataFrame(model.predict(n_periods= len(test_set), X = test_set,m = 7).rename("prediction"),index = test_set.index)
    predictions.index = test_set.index
    predictions['actual'] = test_set[f"{key}"]
    predictions['MAPE'] = predictions.apply(lambda x: MAPE(x['actual'],x['prediction']),axis =1)
    predictions
    plt.figure(figsize=(9, 6))
    plt.plot(training_set[f"{key}"])
    plt.plot(predictions['actual'],label= 'actual')
    plt.plot(predictions['prediction'],label= 'prediction')
    plt.savefig(f"Results/SARIMA_MAPE/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_HW_Predictions.png")
    plt.clf()
    predictions.to_csv(f"Results/SARIMA_MAPE/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_HW_Predictions.csv")
    return predictions

def predict_HW_MAPE(name,training_set,test_set,key,type):
    model = ExponentialSmoothing(endog = training_set[f"{key}"],seasonal_periods = 7).fit()
    predictions = pd.DataFrame(model.forecast(steps= len(test_set)).rename('prediction'),index = test_set.index)
    predictions.index = test_set.index
    predictions['actual'] = test_set[f"{key}"]
    predictions['MAPE'] = predictions.apply(lambda x: MAPE(x['actual'],x['prediction']),axis =1)
    predictions
    plt.figure(figsize=(9, 6))
    plt.plot(training_set[f"{key}"])
    plt.plot(predictions['actual'],label= 'actual')
    plt.plot(predictions['prediction'],label= 'prediction')
    plt.savefig(f"Results/HW_MAPE/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_HW_Predictions.png")
    plt.clf()
    predictions.to_csv(f"Results/HW_MAPE/{type}/{name}_{str(test_set.index[0].date())}_{str(test_set.index[-1].date())}_HW_Predictions.csv")
    return predictions #, MAPE(test_set.revenue,predictions)