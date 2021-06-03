import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from deepcrypto.brokers import *

class BinanceBroker(BrokerBase):
    ORDER_TYPE_DICT = {
        OrderTypes.LIMIT_BUY : Client.ORDER_TYPE_LIMIT,
        OrderTypes.LIMIT_ : Client.ORDER_TYPE_STOP_LOSS,
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
        Client.KLINE_INTERVAL_3DAY : 1440 * 3,
        Client.KLINE_INTERVAL_1WEEK : 1440 * 7,
    }

    def __init__(self, api_key, api_secret, testnet=False, **kwargs):
        self.binance_client = Client(api_key, api_secret)
        self.symbol = self.ticker.replace("/", "")
        self.trade_asset, self.balance_asset = self.ticker.split("/")
        super(BinanceBroker, self).__init__(**kwargs)
        
    
    def init_data(self) -> pd.DataFrame:
        data = self.binance_client.get_historical_klines(f"{self.n_bars * self.INTERVAL2MINUTE[self.timeframe]} minutes ago UTC")
        data = np.array(data[:-1], dtype=np.float32)[:, :6]
        data = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        data.index = pd.to_datetime(data["time"]*1000000)
        return data

    def update_new_data_stream(self) -> Tuple[pd.DataFrame, bool]:
        data = self.binance_client.get_klines(self.symbol, self.timeframe, limit=2)
        closed=False
        if data[0][0] != self.data["time"].iloc[-1]:
            closed = True
            new_data_stream = pd.DataFrame(
                [data[0]], 
                columns=["time", "open", "high", "low", "close", "volume"], 
                index=[pd.to_datetime(data[0][0] * 1000000)]
            )
            self.data = self.data.iloc[1:].append(new_data_stream)
        return closed

    def order(self, quantity, type, price, **kwargs):
        raise NotImplementedError

    def cancel_all_orders(self):
        raise NotImplementedError

    def update_account_info(self):
        raise NotImplementedError

    def get_position_size(self):
        raise NotImplementedError

    def get_position_side(self):
        raise NotImplementedError

    def get_current_price(self):
        raise NotImplementedError

    def get_cash_left(self):
        raise NotImplementedError