import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool

import warnings

import time
from tqdm import tqdm
import sys

warnings.filterwarnings("ignore")


RAW_DATA_PATH = Path("__file__").resolve().parents[0] / "data" / "01_raw"
PROCESSED_DATA_PATH = Path("__file__").resolve().parents[0] / "data" / "02_processed"
MODEL_OUT_PATH = Path("__file__").resolve().parents[0] / "models"


def print_timer(seconds):
    sys.stdout.write(f"\rElapsed Time: {seconds:.2f} seconds")
    sys.stdout.flush()


def read_and_format_csv(subfolder_path: Path, raw_path: bool = True) -> pd.DataFrame:
    if raw_path:
        path = str(RAW_DATA_PATH / subfolder_path)
        df = pd.read_csv(path)
    else:
        path = str(PROCESSED_DATA_PATH / subfolder_path)
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


df = read_and_format_csv(
    subfolder_path="Information_Technology/AAPL.csv", raw_path=True
)

df["Returns"] = df["Adj Close"].pct_change()
df = df[["Date", "Returns"]].dropna().set_index("Date")

num_args_p, num_args_q = (0, 27), (0, 27)

order_ai_bic = []
total_start = time.time()
total_iterations = (num_args_p[1] - num_args_p[0]) * (num_args_q[1] - num_args_q[0])

with tqdm(total=total_iterations, desc="Fitting ARIMA Models") as pbar:
    for p in range(num_args_p[0], num_args_p[1]):
        # Loop over MA order
        for q in range(num_args_q[0], num_args_q[1]):
            try:
                start_time = time.time()
                # Fit model
                model = ARIMA(df, order=(p, 0, q))
                results = model.fit()
                # Print the model order and the AIC/BIC values
                order_ai_bic.append((p, q, results.aic, results.bic))

                elapsed = time.time() - start_time
                print_timer(elapsed)

            except:
                # Print AIC and BIC as None when fails
                print(p, q, None, None)
            pbar.update(1)
    total_time = time.time() - total_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")


order_df = pd.DataFrame(order_ai_bic, columns=["p", "q", "aic", "bic"])
order_df.to_csv(
    str(MODEL_OUT_PATH / f"arima_order_aic_bic_{num_args_p[0]}_{num_args_p[1]}.csv"),
    index=0,
)
