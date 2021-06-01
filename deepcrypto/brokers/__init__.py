import pandas as pd
import numpy as np

from typing import *
from deepcrypto.backtest import *


class OrderTypes:
    MARKET_BUY = 0
    LIMIT_BUY = 1

    MARKET_SELL = 2
    LIMIT_SELL = 3
    
    MARKET_SL = 4
    LIMIT_SL = 5

    MARKET_TP = 6
    LIMIT_TP = 7
    

class BrokerBase():
    def __init__(self, ticker, timeframe, strategy, config, n_bars, market_order=True):
        self.ticker = ticker
        self.timeframe = timeframe
        self.strategy = strategy
        self.config = config
        self.n_bars = n_bars
        self.market_order = market_order

        self.data = self.init_data()

        self.current_pos = 0

    def init_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def update_new_data_stream(self) -> Tuple[pd.DataFrame, bool]:
        """
        return new data stream as pandas dataframe, and return the bar is closed or not
        """
        raise NotImplementedError

    def order(self, quantity, type, price, **kwargs):
        raise NotImplementedError

    def cancel_all_orders(self):
        raise NotImplementedError

    def get_position_size(self):
        raise NotImplementedError

    def get_current_price(self):
        raise NotImplementedError

    def get_cash_left(self):
        raise NotImplementedError

    def get_portfolio_value(self):
        cprice = self.get_current_price()
        cashleft = self.get_cash_left()
        position_size = self.get_position_size()

        return cprice * position_size + cashleft, cprice, cashleft, position_size
        

    def trade_step(self):
        res = self.strategy(self.data, self.config).iloc[-1]
        target_pos = order_logic(self.current_pos, res["enter_long"], res["enter_short"], res["close_long"], res["close_short"])
        if target_pos != self.current_pos:
            
            target_amount = res["bet"] * self.portfolio_value / self.price - self.position_size * self.current_pos
            order_size, order_side = np.abs(target_amount), np.sign(target_amount)

            if order_side == -1:
                order_type = OrderTypes.MARKET_SELL if self.market_order else OrderTypes.LIMIT_SELL
            if order_side == 1:
                order_type = OrderTypes.MARKET_BUY if self.market_order else OrderTypes.LIMIT_SELL
            
            self.cancel_all_orders()
            
            self.order(order_size, order_type, self.price)

            if not target_pos == 0:
                if not res["stop_loss"] != np.inf:
                    self.order(order_size, order_type, (self.price * (1 - res["stop_loss"] * target_pos)))
                if not res["take_profit"] != np.inf:
                    self.order(order_size, order_type, (self.price * (1 + res["take_profit"] * target_pos)))
            

    def main(self):
        _, closed = self.update_new_data_stream()
        self.portfolio_value, self.price, self.cash_left, self.position_size = self.get_portfolio_value()
        if closed:
            self.trade_step()
        



    
        

    