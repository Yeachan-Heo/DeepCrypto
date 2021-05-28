from deepcrypto.data_utils.crawlers.binance_crawler import read_binance_data
from deepcrypto.backtest import run_backtest_df, BacktestAccessor
from sklearn.model_selection import ParameterGrid
import ray, os, datetime, psutil
import quantstats as qs
import pandas as pd
import warnings
import tqdm
import time


def generate_backtest_fn(metric_fn):
    @ray.remote
    def backtest_fn(df, config, strategy, cnt):
        warnings.filterwarnings("ignore")
        df = df.copy()
        df = strategy(df, config)
        order_df, port_df = run_backtest_df(df, log_time=False)
        return metric_fn(order_df, port_df), config, cnt
    return backtest_fn


class OptimizerBase:
    def __init__(self, data, strategy, config_dict, metric_fn, result_dir="./deepcrypto_results", strategy_name="strategy", n_cores=None, total_steps=1):
        self.data = data
        self.data = self.data.backtest.add_defaults()

        self.strategy = strategy
        self.config_dict = config_dict
        self.result_dir = result_dir
        self.total_steps = total_steps

        self.strategy_name = strategy_name

        self.result = [] # dict of config + metrics
        self.n_cores = n_cores if n_cores is not None else psutil.get_cpu_count()
        self.backtest_fn = generate_backtest_fn(metric_fn)

        self.process_queue = []
        self.cnt = -1

    def sample(self) -> tuple:
        self.cnt += 1

    def save_result(self, config, result, cnt):
        if ((not config is None) and (not result is None)):
            dict_ = dict()
            
            dict_.update(config)
            dict_.update(result)
            dict_.update({"index" : cnt})

            self.result.append(dict_)

    def update(self, config, result) -> None:
        return
        
    def optimize(self):
        t = time.time()
        config, result, done = None, None, False
        for i in tqdm.tqdm(range(self.total_steps)):
            for j in range(self.n_cores):
                config, done = self.sample()

                if not done:
                    self.process_queue.append(
                        self.backtest_fn.remote(
                            self.data,
                            config,
                            self.strategy,
                            self.cnt
                        )
                    )

                if len(self.process_queue) > self.n_cores:
                    result, config, cnt = ray.get(self.process_queue[0])
                    
                    self.save_result(config, result, cnt)

                    self.update(config, result)
                    self.process_queue = self.process_queue[1:]

                    if done:
                        for p in self.process_queue:
                            result, config, cnt = ray.get(p)
                            self.save_result(config, result, cnt)
                            
            
            if done: break
        print(f"optimization for {self.cnt} steps completed in {time.time() - t} seconds")

    def save_results(self):
        result_path = os.path.join(self.result_dir, self.strategy_name)
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        result_path = os.path.join(result_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv")
        
        self.result_df.to_csv(result_path)

    def get_best_result(self, by, mode_max=True):
        index = self.result_df[by].argmax() if mode_max else self.result_df[by].argmin()
        ret = self.result_df.iloc[index].to_dict()
        return {key : int(val) if int(val) == val else val for key, val in ret.items()}

    @property
    def result_df(self):
        if hasattr(self, "_result_df"):
            return self._result_df
        self._result_df = pd.DataFrame(self.result)
        self.result.clear()
        return self._result_df


class BruteForceOptimizer(OptimizerBase):
    def __init__(self, **kwargs):
        super(BruteForceOptimizer, self).__init__(**kwargs)
        self.grid = ParameterGrid(self.config_dict)
        self.grid_length = len(self.grid) - 1
        self.total_steps = ((self.grid_length + 1) // self.n_cores) + 1
    
    def sample(self) -> tuple:
        super().sample()
        try:
            return self.grid[self.cnt], False
        except: 
            return None, True


class GeneticOptimizer(OptimizerBase):
    

def test():

    ray.init()

    def crossover_strategy(df, config):
        df["stop_loss"] = config["stop_loss"]
        df["take_profit"] = config["take_profit"]
        df["time_cut"] = config["time_cut"]
        df["trade_cost"] = 0.001

        signal = df.close.ewm(config["ma1"]).mean() > df.close.ewm(config["ma2"]).mean()
        signal = signal & (signal.diff() == 1)

        close_signal = df.close.ewm(config["ma2"]).mean() > df.close.ewm(config["ma1"]).mean()
        close_signal = close_signal & (close_signal.diff() == 1)

        df["enter_long"] = signal
        df["close_long"] = close_signal
        return df

    def metric_fn(order_df, port_df):
        return {"sharpe" : qs.stats.sharpe(port_df["portfolio_value"].resample("1D").last())}

    config_dict = {
        "ma1" : [5, 10, 20, 50, 200, 250, 500],
        "ma2" : [5, 10, 20, 50, 200, 250, 500],
        "time_cut" : [5, 10, 20, 50, 200, 250, 500],
        "stop_loss" : [0.02, 0.05, 0.07, 0.1, 0.15, 0.2],
        "take_profit" : [0.05, 0.07, 0.1, 0.15, 0.2, 0.5]
    }

    data = read_binance_data("/home/ych/Storage/binance/binance.db", "1H", "BTCUSDT")
    
    optimizer = BruteForceOptimizer(data=data, strategy=crossover_strategy, config_dict=config_dict, metric_fn=metric_fn, result_dir="./deepcrypto_results", strategy_name="crossover_strategy", n_cores=12)

    optimizer.optimize()
    
    optimizer.save_results()


if __name__ == "__main__":
    test()