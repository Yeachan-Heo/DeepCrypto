import pandas as pd
import numpy as np
import ccxt

import websocket, os, sqlite3

from typing import *
from deepcrypto.backtest import *


class OrderTypes:
    MARKET = "MARKET"

    LIMIT = "LIMIT"
    
    MARKET_SL = "STOP_MARKET"
    LIMIT_SL = "STOP_LOSS_LIMIT"

    MARKET_TP = "TAKE_PROFIT_MARKET"
    LIMIT_TP = "TAKE_PROFIT_LIMIT"
    

class OrderSides:
    BUY = 1
    SELL = -1


class CCXTWebsocketBrokerBase():
    def __init__(self, ticker, timeframe, strategy, config, n_bars, market_order=True, time_cut_market=True, stop_loss_market=True, take_profit_market=False, log_db_path="./log.sqlite3"):
        self.ticker = ticker
        self.trade_asset, self.balance_asset = ticker.split("/")
        self.symbol = ticker.replace("/", "")

        self.timeframe = timeframe
        self.strategy = strategy
        self.config = config
        self.n_bars = n_bars
        self.market_order = market_order

        self.data = self.init_data()

        self.position_side = 0

        flag = os.path.exists(log_db_path)
        self.log_db = sqlite3.connect(log_db_path)

        self.stop_loss_market = stop_loss_market
        self.take_profit_market = take_profit_market
        self.time_cut_market = time_cut_market

        if not flag:
            self.log_db.execute(
                """CREATE TABLE LOG (
                    timestamp int,
                    portfolio_value float,
                    price float,
                    cash_left float,
                    position_size float,
                    position_side int
                )""")
            self.log_db.commit()

        self.n_bars_from_last_trade = 0 

    def add_to_data(self, ohlcv):
        self.data = self.data.append(pd.DataFrame(ohlcv, [ohlcv["time"]]))

    def preprocess_msg(self, msg):
        ohlcv = {
            "time" : 0,
            "open" : 0,
            "high" : 0,
            "low" : 0,
            "close" : 0,
            "volume" : 0
         } 
        closed = False
        return ohlcv, closed

    def fetch_balance(self, asset):
        raise NotImplementedError

    def fetch_position(self):
        raise NotImplementedError
        # return position_size, position_side

    def get_portfolio_value(self):

        cashleft = self.fetch_balance(self.balance_asset)
        position_size, position_side = self.fetch_position()
        
        self.portfolio_value, self.price, self.cash_left, self.position_size, self.position_side = \
             self.data.iloc[-1]["close"] * position_size + cashleft, self.data.iloc[-1]["close"], cashleft, position_size, position_side
    

    def order(self, quantity, type, side, price):
        raise NotImplementedError

    def cancel_all_orders(self):
        raise NotImplementedError

    def trade_step(self):
        res = self.strategy(self.data, self.config).iloc[-1]
        target_pos = order_logic(self.position_side, res["enter_long"], res["enter_short"], res["close_long"], res["close_short"])
        
        self.n_bars_from_last_trade += 1

        if ((self.n_bars_from_last_trade >= res["time_cut"]) & self.position_side):
            self.order(
                self.position_size,
                OrderTypes.LIMIT if not self.time_cut_market else OrderTypes.MARKET,
                -self.position_side,
                self.price
            )

        if target_pos != self.position_side:
            
            target_amount = res["bet"] * self.portfolio_value / self.price - self.position_size * self.position_side
            order_size, order_side = np.abs(target_amount), np.sign(target_amount)
            
            self.cancel_all_orders()
            
            self.order(order_size, OrderTypes.LIMIT if not self.market_order else OrderTypes.MARKET, order_side, self.price)
            self.n_bars_from_last_trade = 0

            if not target_pos == 0:
                order_size = target_amount
                order_side = -order_side

                if not res["stop_loss"] != np.inf:
                    order_type = OrderTypes.MARKET_SL if self.stop_loss_market else OrderTypes.LIMIT_SL
                    self.order(order_size, order_type, order_side, (self.price * (1 - res["stop_loss"] * target_pos)))
                
                if not res["take_profit"] != np.inf:
                    order_type = OrderTypes.MARKET_TP if self.stop_loss_market else OrderTypes.LIMIT_TP
                    self.order(order_size, order_type, order_side, (self.price * (1 + res["take_profit"] * target_pos)))


    def on_message(self, msg):
        ohlcv, closed = self.preprocess_msg(msg)
        
        if closed:
            self.add_to_data(ohlcv)
            self.trade_step()
            

