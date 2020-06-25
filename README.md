Time series forecasting
=============================

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->
[![GitHub contributors](https://img.shields.io/github/contributors/scottydocs/README-template.md)](https://github.com/MBoubeta/ProgrammingAssignment2/contributors)
[![GitHub stars](https://img.shields.io/github/stars/jonsn0w/hyde.svg?style=social)](https://github.com/MBoubeta/ProgrammingAssignment2/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/scottydocs/README-template.md?style=social)](https://github.com/MBoubeta/ProgrammingAssignment2/fork)
[![GitHub watchers](https://img.shields.io/github/watchers/jonsn0w/hyde.svg?style=social)](https://github.com/MBoubeta/ProgrammingAssignment2/watchers)

<!---![Twitter Follow](https://img.shields.io/twitter/follow/scottydocs?style=social)--->

Time series forecasting provides ML time series models to predict the future of a target variable.

Specifically, three models are considered: autoregressive integrated moving average (ARIMA), Prophet and  Long Short-Term Memory (LSTM). 

Project Organization
-----------------------------

    ├── config                       <- Configuration environment.
    ├── data
    |
    ├── docs                         <- Docs and references.
    │
    ├── models                       <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks                    <- Jupyter notebooks. 
    │
    ├── src                          <- Source code for use in this project.
    │   ├── __init__.py              <- Makes src a Python module.
    │   │
    │   ├── data                     <- Scripts to download or generate data.
    │   │   └── data_wrangling.py
    │   │
    │   ├── deployment               <- Scripts to deploy the model.
    │   │   └── deployment.py
    │   │    
    │   ├── feature_engineering      <- Scripts to turn raw data into features for modeling.
    │   │   └── features.py
    │   │
    │   ├── models                   <- Scripts to train models and then use trained models to make predictions.                
    │   │   ├── predict.py
    |   |   ├── predict_process.py
    |   |   ├── training.py
    │   │   └── training_process.py
    │   │
    │   └── utils                    <- Scripts with auxiliary functions.
    │   |    └── metrics.py
    │   │
    │   └── visualization            <- Scripts to create exploratory and results oriented visualizations.
    │       └── plots.py
    │  
    ├── test                         <- test files
    |
    ├── LICENSE
    │
    ├── execute_predict_process.py   <- Main function used to get predictions from the trained model.
    │
    ├── execute_training_process.py  <- Main function to train the models.
    |
    ├── Makefile                     <- Makefile with commands like `make data` or `make train`.
    ├── README.md                    <- The top-level README for developers using this project.
    ├── requirements.txt             <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py                     <- makes project pip installable (pip install -e .) so src can be imported.
    |
    ├── tox.ini                      <- Python style guidance.
    |
    └── .pylintrc                    <- tox file with settings for running tox; see tox.testrun.org


--------

