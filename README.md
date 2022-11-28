# Datascience

## Brief Explanation

    I tried using two models, usually I would also go to try tensorflow time series model but the time was not enough for that.
    We can see that the trend and seasonality differs between items, so in an ideal environment for each item there would be a different use case. For the sake of this task There is a folder which is results. It contains the Holt_winters and SARIMA results for the 3 months predicition (Jan, Feb and March of 2022).
    The other two folders are the same instead the test and training set were divided from the existing dataset to calculate the mape.
    The test size for Daily was 31 days. 8 weeks for Weekly and 2 Months for monthly.

## Test run

    To run the Main.ipynb please install the corresponding libraries in requirements.txt by the running the following command.
    The Overall MAPE per item per Day, Week and Month are the outputs in the last 6 cells of the notebook.

```bash
pip install requirements.txt
```