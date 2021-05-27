from deepcrypto.backtest import run_backtest_df, BacktestAccessor
from sklearn.model_selection import ParameterGrid
import ray, os, datetime, psutil
import pandas as pd
import tqdm

def generate_backtest_fn(metric_fn):
    @ray.remote
    def backtest_fn(config, df, strategy, cnt):
        df = strategy(df, config)
        order_df, port_df = run_backtest_df(df)
        return metric_fn(order_df, port_df), config, cnt
    return backtest_fn


class OptimizerBase:
    def __init__(self, data, strategy, config_dict, metric_fn, result_dir="./deepcrypto_results", strategy_name="strategy", n_cores=None):
        self.data = data
        self.data = self.data.backtest.add_defaults()

        self.strategy = strategy
        self.config_dict = config_dict
        self.result_dir = result_dir

        self.strategy_name = strategy_name

        self.result = [] # dict of config + metrics
        self.n_cores = n_cores if n_cores is not None else psutil.get_cpu_count()
        self.backtest_fn = generate_backtest_fn(metric_fn)

        self.process_queue = []
        self.cnt = -1

    def sugesstion_logic(self, config, result) -> tuple:
        """
        returns new_config, done
        """
        raise NotImplementedError

    def suggest_next(self, config, result, cnt, suggest=False):
        if ((not config is None) and (not result is None)):
            dict_ = dict()
            
            dict_.update(config)
            dict_.update(result)
            dict_.update({"index" : cnt})

            self.result.append(dict_)
        
        self.cnt += 1

        if not suggest:
            return
        
        return self.suggestion_logic(config, result)
        
    def optimize(self):
        config, result, done = None, None, False
        for i in tqdm.tqdm(range(self.total_steps)):
            for j in range(self.n_cores):
                config, done = self.suggest_next(config, result, cnt)

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
                    result, config, cnt = ray.get(self.process_queue[-1])
                    self.process_queue = self.process_queue[:-1]

                    if done:
                        for p in self.process_queue:
                            result, config, cnt = ray.get(p)
                            self.suggest_next(config, result, cnt, suggest=False)
            
            if done: break
        

    def save_results(self):
        if not hasattr(self, "result_df"):
            self.result_df = pd.DataFrame(self.result)
        result_path = os.path.join(self.result_dir, self.strategy_name)
        result_path = os.path.join(result_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv")
        
        self.result_df.to_csv(result_path)


class BruteForceOptimizer(OptimizerBase):
    def __init__(self, **kwargs):
        super(BruteForceOptimizer, self).__init__(**kwargs)
        self.grid = ParameterGrid(self.config_dict)
        self.grid_length = len(self.grid) - 1
        self.total_steps = ((self.grid_length + 1) // self.n_cores) + 1
    
    def sugesstion_logic(self, config, result) -> tuple:
        return self.grid[self.cnt], self.cnt >= self.grid_length
        