import pandas as pd   


class Strategy(object):
    def __init__(self, strategy_config):
        self.config = strategy_config

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

