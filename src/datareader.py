RAW_DATA_PATH = Path("__file__").resolve().parents[1] / "data" / "01_raw"
PROCESSED_DATA_PATH = Path("__file__").resolve().parents[1] / "data" / "02_processed"


def read_and_format_csv(subfolder_path: str, raw_path: bool = True) -> pd.DataFrame:
    if raw_path:
        path = str(RAW_DATA_PATH) + f"/{subfolder_path}"
        df = pd.read_csv(path)
    else:
        path = str(PROCESSED_DATA_PATH) + f"/{subfolder_path}"
        df = pd.read_csv(path)

    df = df.astype(
        {
            col: "float32"
            for col in ["Open", "High", "Low", "Close", "Dividends", "Stock Splits"]
        }
    )
    df["Volume"] = df["Volume"].astype("int32")

    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601", utc="False")
    df["Date"] = df["Date"].dt.tz_convert("America/New_York")
    return df
