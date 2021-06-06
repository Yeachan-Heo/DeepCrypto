import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from deepcrypto.brokers import *

class BinanceSpotBroker(BrokerBase):
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
        Client.KLINE_INTERVAL_3DAY : 1440 * 3,
        Client.KLINE_INTERVAL_1WEEK : 1440 * 7,
    }

    def __init__(self, api_key, api_secret, testnet=False, **kwargs):
        self.binance_client = Client(api_key, api_secret, testnet=testnet)
        
        super(BinanceSpotBroker, self).__init__(**kwargs)

        self.symbol = self.ticker.replace("/", "")
        self.trade_asset, self.balance_asset = self.ticker.split("/")
        self.n_bars_from_last_order = 0
        
        
    
    def init_data(self) -> pd.DataFrame:
        data = self.binance_client.get_historical_klines(
            self.ticker.replace("/", ""), 
            self.timeframe,
            f"{self.n_bars * self.INTERVAL2MINUTE[self.timeframe]} minutes ago UTC")
        data = np.array(data[:-1], dtype=np.float32)[:, :6]
        data = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        data.index = pd.to_datetime(data["time"]*1000000)
        return data

    def update_new_data_stream(self) -> Tuple[pd.DataFrame, bool]:
        data = self.binance_client.get_klines(symbol=self.symbol, interval=self.timeframe, limit=2)
        closed=False
        if data[0][0] != self.data["time"].iloc[-1]:
            closed = True
            new_data_stream = pd.DataFrame(
                [np.array(data[0][:6], dtype=np.float32)], 
                columns=["time", "open", "high", "low", "close", "volume"], 
                index=[pd.to_datetime(data[0][0] * 1000000)]
            )
            self.data = self.data.iloc[1:].append(new_data_stream)
        return closed

    def order(self, quantity, order_type, side, price, **kwargs):
        self.binance_client.create_order(
            symbol=self.symbol,
            type=self.ORDER_TYPE_DICT[order_type],
            side=SIDE_SELL if side == OrderSides.SELL else SIDE_BUY,
            quantity=quantity,
            price=price,
            stop_price=price
        )

    def cancel_all_orders(self):
        self.binance_client.cancel_order(symbol=self.symbol)

    def update_account_info(self):
        cash_left = self.binance_client.get_asset_balance(asset=self.balance_asset)
        position_size = self.binance_client.get_asset_balance(asset=self.trade_asset) 
        position_size = position_size if position_size is not None else 0
        position_side = np.sign(position_size) if position_size is not None else 0
        
        return cash_left + position_size * self.price, cash_left, position_size, position_side

    def get_current_price(self):
        return float(self.binance_client.get_recent_trades(symbol=self.symbol, limit=1)[0]["price"])