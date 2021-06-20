from deepcrypto.backtest import *
from quantstats.stats import *
import quantstats as qs
import matplotlib.pyplot as plt

import os, json

def _calc_from_order_df(order_df, label="bothside"):
    a = {}
    a[f"{label}_win_rate"] = order_df.realized[order_df.realized >= 0].count()/order_df.realized.count() * 100
    a[f"{label}_profit_factor"] = -order_df.realized[order_df.realized >= 0].mean()/order_df.realized[order_df.realized < 0].mean()
    a[f"{label}_tpi"] = a[f"{label}_win_rate"] * (1 + a[f"{label}_profit_factor"])
    a[f"{label}_total_profit"] = order_df.realized.sum()
    a[f"{label}_avg_holding_bars"] = order_df.hold_bars.mean()
    a[f"{label}_total_trades"] = len(order_df.index)
    return a

def calc_from_order_df(order_df):
    order_df = order_df[order_df.realized != 0]
    s_df = order_df[order_df.prev_side < 0]
    b_df = order_df[order_df.prev_side > 0]

    ret = {}

    ret.update(_calc_from_order_df(s_df, "sellside"))
    ret.update(_calc_from_order_df(b_df, "buyside"))
    ret.update(_calc_from_order_df(order_df))
    return ret

def metric_from_quantstats(portfolio_value_seq, **kwargs):
    metrics = qs.reports.metrics(portfolio_value_seq, display=False, mode="full")
    dict_ = dict(zip([x.replace(" ", "") for x in metrics.index], [x[0] for x in metrics.values]))
    return dict_

def all_metrics(portfolio_df, order_df):
    ret = {}
    ret.update(metric_from_quantstats(portfolio_df["portfolio_value"]))
    ret.update(calc_from_order_df(order_df))
    return ret


def plot_signals(data, order, n=10):
    plt.figure(figsize=(30, 10))
    order = order.copy()
    order["Date"] = order.index
    order = order[data.index[-n]:data.index[-1]]
    plt.plot(data.open[-n:])
    plt.scatter(order.loc[order['desired_side'] ==1 , 'Date'].values,order.loc[order['desired_side'] ==1, 'order_price'].values, label='skitscat', color='green', s=100, marker="^")
    plt.scatter(order.loc[order['desired_side'] ==0 , 'Date'].values,order.loc[order['desired_side'] ==0, 'order_price'].values, label='skitscat', color='black', s=100, marker="X")
    plt.scatter(order.loc[order['desired_side'] ==-1 , 'Date'].values,order.loc[order['desired_side'] ==-1, 'order_price'].values, label='skitscat', color='red', s=100, marker="v")


def save_results(cfg, strategy, df, save_dir, save_name):
    sdir = os.path.join(save_dir, save_name)

    if not os.path.exists(sdir):
        os.makedirs(sdir)

    df = df.backtest.add_defaults()
    order_df, portf_df = strategy(cfg, df).backtest.run()
    metrics = all_metrics(portf_df.resample("1D").last(), order_df)

    with open(os.path.join(sdir, "config.json"), "w") as f:
        f.write(json.dumps(cfg))
    with open(os.path.join(sdir, "metrics.json"), "w") as f:
        f.write(json.dumps(metrics))

    portf_df.to_csv(os.path.join(sdir, "portfolio.csv"))
    order_df.to_csv(os.path.join(sdir, "order.csv"))

    qs.reports.html(portf_df["portfolio_value"], benchmark=portf_df["open"], output=os.path.join(sdir, "report.html"))