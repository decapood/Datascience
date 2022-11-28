import pandas as pd
import numpy as np
#Adjust dataset to fit timeseries forecasting models
def Adjust_DataSet(dataset,frequency,key):
    dataset['date'] = pd.to_datetime(dataset['date'])
    if frequency=="W":
        dataset = dataset.groupby([pd.Grouper(key='date', freq='W-SUN')])[f'{key}'].sum().reset_index()
    elif frequency == "M":
        dataset = dataset.groupby([pd.Grouper(key='date', freq='M')])[f'{key}'].sum().reset_index()
    dataset = dataset.set_index(['date'])
    dataset = dataset.asfreq(f"{frequency}")
    meanValue = dataset[f'{key}'].mean()
    dataset[f'{key}'].fillna(value = meanValue, inplace = True)
    return dataset