# Build a pipeline to transform raw data
import pandas as pd
from pathlib import Path
import yfinance as yf


RAW_DATA_PATH = Path("__file__").resolve().parents[0] / "data" / "01_raw"
PROCESSED_DATA_PATH = Path("__file__").resolve().parents[0] / "data" / "02_processed"

BLUE_CHIP_STOCKS_100 = {
    "Information_Technology": [  # 26% of S&P500
        "MSFT",
        "AAPL",
        "CSCO",
        "INTC",
        "IBM",
        "ORCL",
        "ACN",
        "TXN",
        "ADBE",
        "AMAT",
        "ADI",
        "KLAC",
        "LRCX",
        "NVDA",
        "QCOM",
        "HPQ",
        "ANET",
        "CDNS",
        "SNPS",
        "INTU",
        "AVGO",
        "ADSK",
        "CTSH",
        "FFIV",
        "JNPR",
    ],
    "Health_Care": [  # 13% of S&P500
        "JNJ",
        "PFE",
        "MRK",
        "ABT",
        "UNH",
        "BMY",
        "MDT",
        "AMGN",
        "BDX",
        "BAX",
        "ISRG",
        "ZBH",
        "SYK",
    ],
    "Financials": [  # 13% of S&P500
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "AXP",
        "USB",
        "PNC",
        "TFC",
        "BK",
        "SCHW",
        "AIG",
    ],
    "Consumer_Discretionary": [  # 10% of S&P500
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "TGT",
        "LOW",
        "BBY",
        "DIS",
        "AMZN",
        "EBAY",
    ],
    "Industrials": [  # 8% of S&P500
        "GE",
        "MMM",
        "HON",
        "UNP",
        "CAT",
        "LMT",
        "RTX",
        "UPS",
    ],
    "Communication_Services": [  # 8% of S&P500
        "T",
        "VZ",
        "CMCSA",
        "NFLX",
        "GOOGL",
        "META",
        "DIS",
        "CHTR",
    ],
    "Consumer_Staples": [  # 7% of S&P500
        "PG",
        "KO",
        "PEP",
        "WMT",
        "COST",
        "CL",
        "GIS",
    ],
    "Energy": [  # 5% of S&P500
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
    ],
    "Utilities_": [  # 3% of S&P500
        "NEE",
        "DUK",
        "SO",
    ],
    "Real_Estate": [  # 3% of S&P500
        "AMT",
        "PLD",
        "EQIX",
    ],
    "Materials": [  # 2% of S&P500
        "LIN",
        "APD",
    ],
    "Index_Funds": [
        "^GSPC",
        "^NDX",
    ],
}

BLUE_CHIP_STOCKS_20 = {
    "Information_Technology": [
        "AAPL",
        "MSFT",
        "NVDA",
        "ADBE",
        "CSCO",
    ],
    "Health_Care": [
        "JNJ",
        "UNH",
        "PFE",
    ],
    "Financials": [
        "JPM",
        "BAC",
        "GS",
    ],
    "Consumer_Discretionary": [
        "AMZN",
        "HD",
    ],
    "Industrials": [
        "HON",
        "UNP",
    ],
    "Communication_Services": [
        "GOOGL",
        "META",
    ],
    "Consumer_Staples": ["PG"],
    "Energy": ["XOM"],
    "Utilities_": ["NEE"],
    "Index_Funds": [
        "^GSPC",
        "^NDX",
    ],
}


def get_dataset() -> None:
    """
    Get data for stocks using yfinance yahoo api, and save to csv
    and a concatenated version based on sector, eg, Information_Technology
    """
    for industry, stocks in BLUE_CHIP_STOCKS_20.items():
        print("=" * 10, industry, "=" * 10)
        dataset_location = RAW_DATA_PATH / industry
        dataset_location.mkdir(parents=True, exist_ok=True)
        frames = []
        for stock in stocks:
            stock_parse = yf.download(stock).reset_index()
            stock_parse.columns = stock_parse.columns.get_level_values(0)
            stock_parse["stock_name"] = stock
            stock_parse.to_csv(f"{str(dataset_location)}/{stock}.csv", index=0)
            frames.append(stock_parse)

        industry_frame = pd.concat(frames)
        industry_frame.to_csv(f"{str(PROCESSED_DATA_PATH)}/{industry}.csv", index=0)


get_dataset()
