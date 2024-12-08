from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.metrics as metrics

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf

RAW_DATA_PATH = Path("__file__").absolute().parents[0] / "data" / "01_raw"

print(tf.config.list_physical_devices("GPU"))


def read_and_format_csv(
    subfolder_path: Path,
) -> pd.DataFrame:
    path = str(RAW_DATA_PATH / subfolder_path)
    df = pd.read_csv(path)

    df = df.astype(
        {
            col: "float32"
            for col in [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
            ]
        }
    )
    df["stock_name"] = df["stock_name"].astype("str")
    df["Volume"] = df["Volume"].astype("int32")

    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601", utc="True")
    df["Date"] = pd.to_datetime(df["Date"].dt.date)
    return df


df = read_and_format_csv(subfolder_path="Information_Technology/AAPL.csv")
df.dropna(inplace=True)
df.sort_values(by="Date", inplace=True)

df_deep = df[["Date", "Adj Close"]].set_index("Date")

scaler = MinMaxScaler(feature_range=(0, 1))
df_deep["Adj Close"] = scaler.fit_transform(df_deep[["Adj Close"]])

sequence_length = 10


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i : (i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


X, y = create_sequences(df_deep["Adj Close"].values, sequence_length)


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


param_grid = {
    "lstm_units": [32, 50, 64],
    "dropout_rate": [0.1, 0.2, 0.3],
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [16, 32],
}

# Create parameter combinations
param_combinations = list(ParameterGrid(param_grid))

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)


# Function to create and compile model with given parameters
def create_model(lstm_units, dropout_rate, learning_rate):
    model = Sequential(
        [
            LSTM(
                lstm_units,
                activation="relu",
                return_sequences=True,
                input_shape=(sequence_length, 1),
            ),
            Dropout(dropout_rate),
            LSTM(lstm_units, activation="relu"),
            Dropout(dropout_rate),
            Dense(1),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            metrics.RootMeanSquaredError(),
            metrics.MeanAbsolutePercentageError(),
            metrics.MeanAbsoluteError(),
        ],
    )
    return model


# Grid search implementation
best_loss = float("inf")
best_params = None
history_dict = {}

for params in param_combinations:
    """GridSearch approach with trying all range of hyperparmeter compinations
    """
    print(f"Training with parameters: {params}")

    # Create and train model
    model = create_model(
        params["lstm_units"], params["dropout_rate"], params["learning_rate"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=params["batch_size"],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Evaluate model
    val_loss = min(history.history["val_loss"])
    history_dict[str(params)] = history.history

    # Update best parameters if needed
    if val_loss < best_loss:
        best_loss = val_loss
        best_params = params

print(f"\nBest parameters: {best_params}")
print(f"Best validation loss: {best_loss}")
