## Navigating this project

<p>I undertsant this project has a lot of files and maybe hard to navigate,<br> therefore this is a summary of where all files are located.</p>

📁 <font color='orange'>**data/**</font><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📁 01_raw/: Contains raw data files from API<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📁 Information_Technology/: Contains AAPL stock data<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📁 02_processed/: Processed and cleaned data files<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; AAPL_processed.csv: Processed Apple stock data<br>

📁 <font color='orange'>**models/**</font><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; optimal_arima_model.py: ARIMA model optimization<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; optimal_deep_learning.py: Deep learning GridSearch optimization<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; optimal_deep_learning_keras_tuner.py: Keras tuner implementation<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📁 ARIMA_aic_bic/: ARIMA model gridsearch eval results<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📁 <font color='orange'>**model_outputs/**</font>: Storage for model results<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📁 deep_tuning_params/: Hyperparameter tuning results for LSTM<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📁 post_hyperparameter_tuning/: Post-tuning model configurations<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📁 pre_hyperparam_tuning/: Initial model configurations<br>

📁 <font color='orange'>**notebooks/**</font><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <font color='orange'>**Main Notebooks:**</font><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 01_arima_modeling.ipynb: ARIMA model development<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 02_lstm_modelling.ipynb: LSTM model development<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📁 extra_notebooks/:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 01_check_data.ipynb: Data validation and checks<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 03_auto_arima.ipynb: Automated ARIMA implementation<br>

📁 <font color='orange'>**outputs/**</font><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📁 figures/: Generated visualizations and plots<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Various PNG files for time series analysis, forecasts, and diagnostics<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📁 reports/: Project documentation and reports<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Final report and documentation in DOCX format<br>
📁 <font color='orange'>**src/**</font><br>
main.py: Data extracter from yfinance API<br>

