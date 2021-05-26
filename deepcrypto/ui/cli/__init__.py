from deepcrypto.portfolio_analysis import save_results
from deepcrypto.backtest import *

import argparse, sqlite3, os, json, yaml
import pandas as pd


VALID_DATA_FORMAT = ["sqlite3", "csv"]
COLUMN_CONVERSION_DICT = {
    "t" : "timestamp",
    "o" : "open",
    "h" : "high",
    "l" : "low",
    "c" : "close",
    "v" : "volume" 
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_format", default="sqlite3", type=str, help="db or csv")
    parser.add_argument("--data_path", default=None, type=str, help="data (db or csv) path")
    parser.add_argument("--table_name", default=None, type=str, help="table name like BTCUSDT_1H, required only on db mode")
    parser.add_argument("--columns", default="tohlcv", type=str, help="columns")
    parser.add_argument("--strategy", default=None, type=str, help="strategy file path")
    parser.add_argument("--output_path", default=None, type=str, help="output path")
    parser.add_argument("--strategy_name", default=None, type=str, help="strategy name")
    parser.add_argument("--config_path", default=None, type=str, help="path to config")
    return parser.parse_args()

def transform_columns(columns):
    for column in columns:
        if not column in COLUMN_CONVERSION_DICT.keys():
            raise ValueError(f"column {column} is not valid. --columns would be such as ohlcvt, tochlv.. etc")
    return [COLUMN_CONVERSION_DICT[column] for column in columns]

def read_data(data_format, data_path, table_name, columns):
    columns = transform_columns(columns)
    
    if not os.path.exists(data_path):
        raise ValueError(f"data not exists: {data_path}")

    if data_format == "sqlite3":
        db = sqlite3.connect(data_path)
        data = db.execute(f"SELECT * FROM {table_name}").fetchall()
        data = pd.DataFrame(data)
        data.columns = columns
    
    elif data_format == "csv":
        data = pd.read_csv(data_path)
        data.columns = columns

    data.index = pd.to_datetime(data["timestamp"] * 1000)

    return data

def read_strategy(strategy_path):
    strategy_str = ""
    with open(strategy_path, "r") as f:
        return f.read()
    return strategy_str

#    strategy_str = (f"""{strategy_str}\n        return data""")



def read_config(config_path):
    config_format = config_path.split(".")[-1]
    if config_format not in ["json", "yaml"]:
        raise ValueError(f"invalid format {config_format}")
    with open(config_path, "r") as f:
        config = json.load(f) if config_format == "json" else yaml.load(f)
    return config

def main():
    # try:
        args = parse_args()
        data = read_data(args.data_format, args.data_path, args.table_name, args.columns)
        strategy = read_strategy(args.strategy)
        config = read_config(args.config_path)
        exec(strategy)
        save_results(config, lambda s, d: d, data, args.output_path, args.strategy_name)
            
    # except Exception as e:
    #     print("Durning the backtesting, an exception has occured: \n {}".format(e))
    #     print("exiting...")
    #     exit()
        

    

        