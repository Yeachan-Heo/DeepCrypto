import pandas as pd
import numpy as np
import sqlite3

from typing import *
from deepcrypto.backtest import *


class OrderTypes:
    MARKET = 0

    LIMIT = 0
    
    MARKET_SL = 4
    LIMIT_SL = 5

    MARKET_TP = 6
    LIMIT_TP = 7
    
class OrderSides:
    BUY = 1
    SELL = -1

class BrokerBase():
    def __init__(self, ticker, timeframe, strategy, config, n_bars, market_order=True, stoploss_market=True, takeprofit_market=False, log_db_path="./log.sqlite3"):
        self.ticker = ticker
        self.timeframe = timeframe
        self.strategy = strategy
        self.config = config
        self.n_bars = n_bars
        self.market_order = market_order

        self.data = self.init_data()

        self.position_side = 0

        flag = os.path.exists(log_db_path)
        self.log_db = sqlite3.connect(log_db_path)

        self.stoploss_market = stoploss_market
        self.takeprofit_market = takeprofit_market

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

    def init_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def update_new_data_stream(self) -> Tuple[pd.DataFrame, bool]:
        """
        return new data stream as pandas dataframe, and return the bar is closed or not
        """
        raise NotImplementedError

    def order(self, quantity, type, side, price, **kwargs):
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

    def get_portfolio_value(self):
        self.update_account_info()

        cprice = self.get_current_price()
        cashleft = self.get_cash_left()
        position_size = self.get_position_size()
        position_side = self.get_position_side()

        return cprice * position_size + cashleft, cprice, cashleft, position_size, position_side
        

    def trade_step(self):
        res = self.strategy(self.data, self.config).iloc[-1]
        target_pos = order_logic(self.position_side, res["enter_long"], res["enter_short"], res["close_long"], res["close_short"])
        
        if target_pos != self.position_side:
            
            target_amount = res["bet"] * self.portfolio_value / self.price - self.position_size * self.position_side
            order_size, order_side = np.abs(target_amount), np.sign(target_amount)
            
            self.cancel_all_orders()
            
            self.order(order_size, OrderTypes.LIMIT if not self.market_order else OrderTypes.MARKET, order_side, self.price)

            if not target_pos == 0:
                if not res["stop_loss"] != np.inf:
                    order_type = OrderTypes.MARKET_SL if self.stoploss_market else OrderTypes.LIMIT_SL
                    self.order(order_size, order_type, order_side, (self.price * (1 - res["stop_loss"] * target_pos)))
                if not res["take_profit"] != np.inf:
                    order_type = OrderTypes.MARKET_TP if self.stoploss_market else OrderTypes.LIMIT_TP
                    self.order(order_size, order_type, order_side, (self.price * (1 + res["take_profit"] * target_pos)))
            

    def main(self):
        closed = self.update_new_data_stream()
        self.portfolio_value, self.price, self.cash_left, self.position_size, self.position_side = self.get_portfolio_value()
        
        self.log_db.execute(
            f"""
            INSERT INTO LOG VALUES (
                {self.portfolio_value},
                {self.price},
                {self.cash_left},
                {self.position_size},
                {self.position_side}
            )
            """
        )
        self.log_db.commit()

        if closed:
            self.trade_step()
        


    
        

    