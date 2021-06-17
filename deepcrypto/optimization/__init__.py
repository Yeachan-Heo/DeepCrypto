from deepcrypto.data_utils.crawlers.binance_crawler import read_binance_data
from deepcrypto.backtest import run_backtest_df, BacktestAccessor
from sklearn.model_selection import ParameterGrid
import ray, os, datetime, psutil
import matplotlib.pyplot as plt
import quantstats as qs
import pandas as pd
import numpy as np
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
        
        try: metric = metric_fn(order_df, port_df)
        except: metric = None
        
        return metric, config, cnt, order_df, port_df
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
                    result, config, cnt, _, _ = ray.get(self.process_queue[0])
                    
                    if not result is None:
                        self.save_result(config, result, cnt)
                        self.update(config, result)

                    self.process_queue = self.process_queue[1:]

                    if done:
                        for p in self.process_queue:
                            result, config, cnt = ray.get(p)
                            if not result is None:
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


def do_forward_testing(data, optimizer_cls, n_chunks=4, best_metric_name="metric", mode_max=True, **kwargs):
    data_chunk_len = len(data.index) // n_chunks
    
    order_df_lst, port_df_lst = [], []
    return_arr = None
    initial_cash = 0

    for i in range(1, n_chunks):
        data_train = data[:(i)*data_chunk_len]
        
        if data_train.empty:
            raise ValueError

        data_test = data[(i)*data_chunk_len:np.clip((i+1)*data_chunk_len, 0, len(data.index))]

        optim = optimizer_cls(data=data_train, **kwargs)
        optim.optimize()
        best_config = optim.get_best_result(best_metric_name, mode_max)

        if not data_test.empty:
            _, _, _, order_df, port_df = ray.get(
                optim.backtest_fn.remote(data_test, best_config, optim.strategy, 0))
            order_df_lst.append(order_df)
            port_df_lst.append(port_df)

    for port_df in port_df_lst:
        if return_arr is None:
            initial_cash = port_df["portfolio_value"].iloc[0]
            return_arr = port_df["portfolio_value"] / initial_cash
        return_arr = return_arr.append(port_df["portfolio_value"] / initial_cash * return_arr.iloc[-1])
        return_arr = return_arr[~return_arr.index.duplicated()]
        return_arr = return_arr.sort_index()

    return return_arr, best_config, order_df_lst, port_df_lst, optim.result_df


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
        return {"metric" : qs.stats.sharpe(port_df["portfolio_value"].resample("1D").last())}

    config_dict = {
        "ma1" : [5, 20, 50, 200, 500],
        "ma2" : [5, 20, 50, 200, 500],
        "time_cut" : [5, 20, 50, 200],
        "stop_loss" : [0.02, 0.05, 0.1],
        "take_profit" : [0.1, 0.2, 0.5]
    }

    data = read_binance_data("/home/ych/Storage/binance/binance.db", "1H", "BTCUSDT")
    data = data.backtest.add_defaults()

    return_arr, best_config, order_df_lst, port_df_lst = do_forward_testing(
        data=data, 
        optimizer_cls=BruteForceOptimizer,
        strategy=crossover_strategy, 
        config_dict=config_dict, 
        metric_fn=metric_fn, 
        result_dir="./deepcrypto_results", 
        strategy_name="crossover_strategy", 
        n_cores=12
    )
    print(return_arr)
    for i, port_df in enumerate(port_df_lst):
        qs.reports.html(port_df["portfolio_value"], output=f"./samples/sample{i}.html")

    return_arr.plot()

    plt.savefig("./sample.png")


if __name__ == "__main__":
    test()