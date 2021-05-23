import os
import sqlite3
import pandas as pd
import tqdm

def resample(df : pd.DataFrame, freq):
    df_ohlcv = df[["open", "high", "low", "close", "volume"]]

    logic = {
		'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
	}

    df_ohlcv = df_ohlcv.resample(freq).apply(logic)

    df_timestamp = pd.DataFrame(columns=["time"], index=df_ohlcv.index)
    df_timestamp["time"] = [x.timestamp() for x in df_ohlcv.index]
    df = pd.concat([df_timestamp, df_ohlcv], axis=1).ffill().dropna()

    return df


def to_database(path, timeframes):
	db_path = os.path.join(path, "bitfinex.db")
	csv_path = os.path.join(path, "csv")

	os.system(f"rm -rf {db_path}")
	db = sqlite3.connect(db_path)

	fnames = os.listdir(csv_path)
	tickers = [x[:-4].upper() for x in sorted(fnames)]

	sql_create_table = """
	CREATE TABLE {ticker}_{timeframe} 
		(
			timestamp int, 
			open float, 
			close float, 
			high float, 
			low float,
			volume float
		)
	"""

	sql_insert_table = """
	INSERT INTO {ticker}_{timeframe}
		VALUES {values}
	"""
		
	for ticker in tickers:
		df_dict = {}

		fpath = os.path.join(csv_path, ticker.lower() + ".csv")
		df = pd.read_csv(fpath)
		df["time"] = df["time"] * 1000000
		df.index = pd.to_datetime(df["time"])

		for timeframe in timeframes:
			db.execute(sql_create_table.format(ticker="_" + ticker, timeframe=timeframe))

			if timeframe == "1T":
				df_dict[timeframe] = df
			else:
				df_dict[timeframe] = resample(df, timeframe)
		
		for freq, df in df_dict.items():

			if df.empty:
				continue
			
			df = df[["time", "open", "close", "high", "low", "volume"]]
			for row in tqdm.tqdm(df.values):
				row = tuple(row)
				db.execute(
					sql_insert_table.format(
						ticker="_" + ticker, 
						timeframe=freq, 
						values=row
					)
				)
		
	db.commit()
	db.close()

	
def read_bitfinex_data(db_path, timeframe, ticker):
	db = sqlite3.connect(db_path)

	data = db.execute(f"SELECT * FROM _{ticker}_{timeframe}")
	data = pd.DataFrame(data.fetchall(), columns=["time", "open", "close", "high", "low", "volume"])
	data.index = pd.to_datetime(data["time"] * (1000000 * 1000 if not timeframe == "1T" else 1))
	del data["time"]
	
	return data


if __name__ == "__main__":
	to_database("/home/ych/Storage/bitfinex/", ["1T", "3T", "5T", "10T", "15T", "30T", "1H", "2H", "4H", "1D"])