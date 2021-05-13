from binance.client import Client
import pandas as pd
import numpy as np
import sqlite3, os
import datetime
import tqdm


COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
DEFAULT_START_DATE = "01 Jan 2017"


def download(client, ticker, from_date, to_date):
    data = client.get_historical_klines(ticker, client.KLINE_INTERVAL_1MINUTE, from_date, to_date)
    df = pd.DataFrame(data, columns=COLUMNS)[COLUMNS[:6]]

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["timestamp"] *= 1000000
    df.index = pd.to_datetime(df["timestamp"])

    print(df.head())
    print(df["open"] == df["low"])

    return df


def resample(df : pd.DataFrame, freq):
    if freq == "1T":
        return df

    df_ohlcv = df[["open", "high", "low", "close", "volume"]]

    logic = {'open': 'first',
             'high': 'max',
             'low': 'min',
             'close': 'last',
             'volume': 'sum'}

    df_ohlcv = df_ohlcv.resample(freq).apply(logic)

    df_timestamp = pd.DataFrame(columns=["timestamp"], index=df_ohlcv.index)
    df_timestamp["timestamp"] = [x.timestamp() * 1000000 for x in df_ohlcv.index]
    df = pd.concat([df_timestamp, df_ohlcv], axis=1).ffill().dropna()

    return df


def crawl(db_path, timeframes, tickers, reset_db, api_key, secret_key):
    flag = False

    if reset_db:
        os.system(f"rm -rf {db_path}")

    if not os.path.exists(db_path):
        flag = True


    db = sqlite3.connect(db_path)

    if flag:
        existing_tickers = []
        db.execute("CREATE TABLE TICKERS (ticker text)")
        db.commit()
    else:
        existing_tickers = [x[0] for x in db.execute("SELECT * FROM TICKERS").fetchall()]

    client = Client(api_key, secret_key)

    end_date = datetime.datetime.now() + datetime.timedelta(-1)
    end_date = end_date.strftime("%d %b %Y")


    EXISTING_START_DATE = None

    if not existing_tickers == []:
        EXISTING_START_DATE = read_binance_data(db_path, "1T", existing_tickers[0]).index[-1].to_pydatetime()
        EXISTING_START_DATE += datetime.timedelta(1)
        EXISTING_START_DATE = EXISTING_START_DATE.strftime("%d %b %Y")

    timeframes.pop(timeframes.index("1T"))

    for ticker in tqdm.tqdm(tickers):

        if not ticker in existing_tickers:
            start_date = DEFAULT_START_DATE
            for freq in set(timeframes + ["1T"]):
                db.execute(f"""
                CREATE TABLE {ticker}_{freq} (
                    timestamp int, 
                    open float, 
                    high float, 
                    low float, 
                    close float, 
                    volume float
                )""")
            db.execute(f"INSERT INTO TICKERS VALUES ('{ticker}')")
            db.commit()
        else:
            start_date = EXISTING_START_DATE

        df_dict = {}
        df = download(client, ticker, start_date, end_date)

        df_dict["1T"] = df

        _df_dict = {freq : resample(df, freq) for freq in timeframes}
        df_dict.update(_df_dict)

        for freq, df in df_dict.items():
            for row in df.values:
                row = tuple(row)
                print(row)
                db.execute(f"INSERT INTO {ticker}_{freq} VALUES {row}")
            db.commit()
    db.close()


def read_binance_data(db_path, timeframe, ticker):
    db = sqlite3.connect(db_path)

    data = db.execute(f"SELECT * FROM {ticker}_{timeframe}")
    data = pd.DataFrame(data.fetchall())

    data.columns = COLUMNS[:6]
    data.index = pd.to_datetime(data["timestamp"] * (1 if timeframe == "1T" else 1000))
    data["open"] = data["open"].replace(0, np.nan).ffill()
    data["high"] = data["high"].replace(0, np.nan).ffill()
    data["low"] = data["low"].replace(0, np.nan).ffill()
    data["close"] = data["close"].replace(0, np.nan).ffill()
    del data["timestamp"]
    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--timeframes", type=list, default=["1T", "3T", "5T", "10T", "15T", "30T", "1H", "2H", "4H", "1D"])

    parser.add_argument("--db_path", help="path to sqlite3 database", type=str,
                        default="/home/ych/Storage/binance/binance.db")

    parser.add_argument("--api_key", type=str, default="gXGFWdsBJy6QBu45lICDbfx3jYVpprAgi2tPU6KTYIQremIYFXgJAz6tYxrb3Wjn")
    parser.add_argument("--secret_key", type=str, default="xuCi2FwbBK4hzRRKaTSbktLUuCoU1cmoINsiu9Owwt910Lcpk7gBL3R4kmEsI2AQ")

    parser.add_argument("--reset_db", type=bool, default=True)
    parser.add_argument("--tickers", type=list, default=[
        "BTCUSDT",
        "ETHUSDT",
        "XRPUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "DOTUSDT",
        "BCHUSDT",
        "LTCUSDT",
    ])

    args = parser.parse_args()

    crawl(args.db_path, args.timeframes, args.tickers, args.reset_db, args.api_key, args.secret_key)

