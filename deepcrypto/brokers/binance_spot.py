import pandas as pd
import numpy as np
import json
from binance.client import Client
from binance.enums import *
from binance.helpers import round_step_size
from deepcrypto.brokers.ws_broker import *


class BinanceSpotBroker(WebsocketBrokerBase):
    ORDER_TYPE_DICT = {
        OrderTypes.LIMIT : Client.ORDER_TYPE_LIMIT,
        OrderTypes.MARKET : Client.ORDER_TYPE_MARKET,
        OrderTypes.MARKET_SL : Client.ORDER_TYPE_STOP_LOSS,
        OrderTypes.MARKET_TP : Client.ORDER_TYPE_TAKE_PROFIT,
        OrderTypes.LIMIT_SL : Client.ORDER_TYPE_STOP_LOSS_LIMIT,
        OrderTypes.LIMIT_TP : Client.ORDER_TYPE_TAKE_PROFIT_LIMIT
    }

    INTERVAL2MINUTE = {
        Client.KLINE_INTERVAL_1MINUTE : 1,
        Client.KLINE_INTERVAL_3MINUTE : 3,
        Client.KLINE_INTERVAL_5MINUTE : 5,
        Client.KLINE_INTERVAL_15MINUTE : 15,
        Client.KLINE_INTERVAL_30MINUTE : 30,
        Client.KLINE_INTERVAL_1HOUR : 60,
        Client.KLINE_INTERVAL_2HOUR : 120,
        Client.KLINE_INTERVAL_4HOUR : 240,
        Client.KLINE_INTERVAL_6HOUR : 360,
        Client.KLINE_INTERVAL_8HOUR : 480,
        Client.KLINE_INTERVAL_12HOUR : 720,
        Client.KLINE_INTERVAL_1DAY : 1440,
    }

    INTERVAL2STRING = {
        Client.KLINE_INTERVAL_1MINUTE : "1m",
        Client.KLINE_INTERVAL_3MINUTE : "3m",
        Client.KLINE_INTERVAL_5MINUTE : "5m",
        Client.KLINE_INTERVAL_15MINUTE : "15m",
        Client.KLINE_INTERVAL_30MINUTE : "30m",
        Client.KLINE_INTERVAL_1HOUR : "1h",
        Client.KLINE_INTERVAL_2HOUR : "2h",
        Client.KLINE_INTERVAL_4HOUR : "4h",
        Client.KLINE_INTERVAL_6HOUR : "6h",
        Client.KLINE_INTERVAL_8HOUR : "8h",
        Client.KLINE_INTERVAL_12HOUR : "12h",
        Client.KLINE_INTERVAL_1DAY : "1d",
    }

    def __init__(self, api_key, api_secret, testnet=False, **kwargs):
        self.binance_client = Client(api_key, api_secret, testnet=testnet)
        
        super(BinanceSpotBroker, self).__init__(**kwargs)
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.INTERVAL2STRING[self.timeframe]}"

        self.symbol = self.ticker.replace("/", "")
        self.trade_asset, self.balance_asset = self.ticker.split("/")
        self.n_bars_from_last_order = 0
        
        info = self.binance_client.get_symbol_info(self.symbol)
        self.stepSize = float(info["filters"][2]["stepSize"])
        self.tickSize = float(info["filters"][0]["tickSize"])
    
    def init_data(self) -> pd.DataFrame:
        data = self.binance_client.get_historical_klines(
            self.ticker.replace("/", ""), 
            self.timeframe,
            f"{self.n_bars * self.INTERVAL2MINUTE[self.timeframe]} minutes ago UTC")
        data = np.array(data[:-1], dtype=np.float32)[:, :6]
        data = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        data["time"] = data["time"] * 1000000
        data.index = pd.to_datetime(data["time"])
        return data

    def preprocess_msg(self, msg):
        msg = json.loads(msg)
        ohlcv = {
            "time" : msg["k"]["t"]*1000000,
            "open" : float(msg["k"]["o"]),
            "high" : float(msg["k"]["h"]),
            "low" : float(msg["k"]["l"]),
            "close" : float(msg["k"]["c"]),
            "volume" : float(msg["k"]["v"])
         } 
        closed = bool(msg["k"]["x"])
        return ohlcv, closed

    def order(self, quantity, order_type, side, price, **kwargs):
        order_kwargs = {
            "symbol" : self.symbol,
            "type" : self.ORDER_TYPE_DICT[order_type],
            "side" : SIDE_SELL if side == OrderSides.SELL else SIDE_BUY,
            "quantity" : round_step_size(quantity, self.stepSize),
        }

        if "LIMIT" in order_type:
            order_kwargs["price"] = round_step_size(price, self.tickSize)
        
        if ("LOSS" in order_type) or ("PROFIT" in order_type):
            order_kwargs["stopPrice"] = round_step_size(price, self.tickSize)
        
        self.binance_client.create_order(**order_kwargs)

    def cancel_all_orders(self):
        orders = self.binance_client.get_open_orders(symbol=self.symbol)
        for order in orders:
            self.binance_client.cancel_order(symbol=self.symbol)#, orderId=order["orderId"])

    def fetch_balance(self, asset):
        balance = self.binance_client.get_asset_balance(asset)
        if balance is None:
            return 0
        return float(balance["free"]) + float(balance["locked"])

    def fetch_position(self):
        position_size = self.fetch_balance(self.trade_asset)
        position_size = 0 if position_size is None else position_size
        position_side = np.sign(position_size)
        return position_size, position_side