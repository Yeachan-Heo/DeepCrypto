import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepCrypto Strategy Builder")

    parser.add_argument("--strategy_dir", help="root directory for strategy", default="./deepcrypto")
    parser.add_argument("--strategy_name", help="a strategy's name", default="Strategy", type=str)

    args = parser.parse_args()


    args.strategy_dir = os.path.join(args.strategy_dir, args.strategy_name)

    if not os.path.exists(args.strategy_dir):
        os.makedirs(args.strategy_dir)



    STRATEGY_TEMPLETE = f"""
    import numpy as np
    import pandas as pd
    import deepcrypto.strategy.utils as utils

    NAME = "{args.name}"

    def strategy(df, config):
        df["enter_long"] = (
            (df.close.rolling(config["ma_long"]).mean() < df.close.rolling(config["ma_short"]).mean())
            & (df.close > df.close.rolling(config["ma_short"]).mean())
        )

        df["close_long"] = (
            df.close < df.close.rolling(config["ma_short"]).mean()
        )

        df["enter_short"] = (
            (df.close.rolling(config["ma_long"]).mean() > df.close.rolling(config["ma_short"]).mean())
            & (df.close < df.close.rolling(config["ma_short"]).mean())
        )

        df["close_short"] = (
            df.close > df.close.rolling(config["ma_short"]).mean()
        )


        df["bet"] = 1
        df["trade_cost"] = 0.001
        df["stop_loss"] = 0.05
        df["take_profit"] = 0.05
        
        return df
    """

    with open(os.path.join(args.strategy_dir, "strategy.py"), "w") as f:
        f.write(STRATEGY_TEMPLETE)


    OPTIMIZE_TEMPLETE = """
    import os
    import ray
    import psutil
    import pandas as pd
    import quantstats as qs


    from deepcrypto.optimization import do_forward_testing, BruteForceOptimizer
    from .strategy import strategy, NAME


    def load_data():
        # implement your data loader here (O, H, L, C, V)
        raise NotImplementedError

    def metric_fn(order_df, port_df):
        portfolio_value = portfolio_value.resample("1D").last()
        daily_return = portfolio_value.pct_change()
        return {"metric" : daily_return.mean() / daily_return.std()}

    def optimize(df):
        return do_forward_testing(
            data = df,
            optimizer_cls = BruteForceOptimizer,
            n_chunks = 10,
            strategy = strategy,
            config_dict = {
                {
                    "ma_short" : [5, 10, 20],
                    "ma_long" : [50, 10, 200]
                }
            }
            metric_fn = metric_fn,
            result_dir="./results",
            strategy_name=NAME,
            n_cores = psutil.cpu_count()
        )


    def save_results(df, return_arr, best_config, order_df_lst, port_df_lst):
        # NotImplementedError
        return

    if __name__ == "__main__":
        df = load_data()
        return_arr, best_config, order_df_lst, port_df_lst, optim_result_df = optimize(df)
        
        

    """


        