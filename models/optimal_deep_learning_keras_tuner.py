from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.metrics as metrics

from keras_tuner import RandomSearch

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf

RAW_DATA_PATH = Path("__file__").absolute().parents[0] / "data" / "01_raw"
MODEL_out = Path("__file__").absolute().parents[0] / "models" / "model_outputs"

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


early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)


def build_model(hp):
    model = Sequential(
        [
            LSTM(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
                return_sequences=True,
                input_shape=(sequence_length, 1),
            ),
            Dropout(hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)),
            LSTM(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            ),
            Dropout(hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            metrics.RootMeanSquaredError(),
            metrics.MeanAbsolutePercentageError(),
            metrics.MeanAbsoluteError(),
        ],
    )

    return model


tuner = RandomSearch(
    build_model,
    objective="loss",
    max_trials=5,
    executions_per_trial=3,
    directory=str(MODEL_out / "deep_tuning_params"),
    project_name="LSTM_tuning",
)

tuner.search(
    x=X_train,
    y=y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

loss = model.evaluate(X_test, y_test)

model.save(str(MODEL_out / "post_hyperparam_tuning" / "keras_tuning_best_LSTM.h5"))
print("Validation loss: ", loss)
