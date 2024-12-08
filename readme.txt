Navigating this project

I undertsant this project has a lot of files and maybe hard to navigate,
therefore this is a summary of where all files are located.

📁 data/
    01_raw/: Contains raw data files from API
        Information_Technology/: Contains AAPL stock data
    02_processed/: Processed and cleaned data files
        AAPL_processed.csv: Processed Apple stock data

📁 models/
    optimal_arima_model.py: ARIMA model optimization
    optimal_deep_learning.py: Deep learning GridSearch optimization
    optimal_deep_learning_keras_tuner.py: Keras tuner implementation

    ARIMA_aic_bic/: ARIMA model implementations
    model_outputs/: Storage for model results
        deep_tuning_params/: Hyperparameter tuning results for LSTM
        post_hyperparameter_tuning/: Post-tuning model configurations
        pre_hyperparam_tuning/: Initial model configurations

📁 notebooks/
    Main Notebooks:
        01_arima_modeling.ipynb: ARIMA model development
        02_lstm_modelling.ipynb: LSTM model development
    extra_notebooks/:
        01_check_data.ipynb: Data validation and checks
        03_auto_arima.ipynb: Automated ARIMA implementation

📁 outputs/
    figures/: Generated visualizations and plots
        Various PNG files for time series analysis, forecasts, and diagnostics
    reports/: Project documentation and reports
        Final report and documentation in DOCX format
📁 src/
    main.py: Data extracter from yfinance API
