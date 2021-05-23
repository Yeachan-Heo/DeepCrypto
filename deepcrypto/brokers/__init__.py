import numpy as np


class BrokerBase:
    broker_metadata = {
        "allow_short" : False
    }
    config = {
        "trade_amount" : None,
    }

    def __init__(self, broker_config, strategy_cls, strategy_config):
        self.config = broker_config
        self.strategy = strategy_cls(strategy_config)

    def order(self, quantity, side):
        raise NotImplementedError

    def order_target_quantity(self, quantity):
        quantity = quantity - self.get_portfolio()["position_size"]
        

    def get_portfolio(self):
        ret = {"position_size" : None, "avg_price" : None, "cash_left" : None}
        raise NotImplementedError

    
        

    