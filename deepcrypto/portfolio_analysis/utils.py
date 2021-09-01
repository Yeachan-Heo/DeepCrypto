from deepcrypto.backtest import *
from quantstats.stats import *
import quantstats as qs
import matplotlib.pyplot as plt

import os, json
import numpy as np
from warnings import filterwarnings


filterwarnings("ignore")


def savediv(a, b):
    try:
        return a / b
    except:
        return np.nan


def get_mdd_simple_interest(x):
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[0] * 100


def get_daily_returns_simple_interest(x):
    daily_ret = x.diff() / x[0]
    return daily_ret.mean() * 100


def get_annual_returns_simple_interest(x, n_trade_days=365):
    return get_daily_returns_simple_interest(x) * n_trade_days


def get_sharpe_simple_interest(x, n_trade_days=365):
    daily_ret = x.diff() / x[0]
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(n_trade_days)
    return sharpe


def get_simple_interest_metrics(portfolio_seq):
    drawdown_days = qs.stats.drawdown_details(portfolio_seq).days
    ret = {
        "annual_returns_percent": get_annual_returns_simple_interest(portfolio_seq),
        "daily_returns_percent": get_daily_returns_simple_interest(portfolio_seq),
        "maximum_drawdown_percent": get_mdd_simple_interest(portfolio_seq),
        "sharpe_ratio": get_sharpe_simple_interest(portfolio_seq),
        "maximum_underwater days": drawdown_days.max(),
        "average_underwater days": drawdown_days.mean(),
    }
    ret["cagr_mdd_ratio"] = (
        -ret["annual_returns_percent"] / ret["maximum_drawdown_percent"]
    )

    return ret


def _calc_from_order_df(order_df, label="bothside"):
    a = {}
    a[f"{label}_win_rate"] = (
        order_df.realized[order_df.realized >= 0].count()
        / order_df.realized.count()
        * 100
    )

    a[f"{label}_realized_fee"] = order_df.fee.sum()

    a[f"{label}_avg_profit"] = order_df.realized[order_df.realized > 0].mean()
    a[f"{label}_avg_loss"] = -order_df.realized[order_df.realized < 0].mean()

    a[f"{label}_max_profit_per_trade"] = order_df.realized.max()
    a[f"{label}_max_loss_per_trade"] = -order_df.realized.min()

    a[f"{label}_total_profit"] = order_df.realized[order_df.realized > 0].sum()
    a[f"{label}_total_loss"] = -order_df.realized[order_df.realized < 0].sum()
    a[f"{label}_total_pnl"] = order_df.realized.sum()

    a[f"{label}_profit_factor"] = savediv(
        float(a[f"{label}_total_profit"]), a[f"{label}_total_loss"]
    )
    a[f"{label}_avg_profit_factor"] = savediv(
        a[f"{label}_avg_profit"], a[f"{label}_avg_loss"]
    )

    a[f"{label}_tpi"] = (
        a[f"{label}_win_rate"] * (1 + a[f"{label}_avg_profit_factor"]) / 100
    )

    a[f"{label}_pnl_stdev"] = order_df.realized.std()

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


def metrics_from_order_df(order_df):
    return calc_from_order_df(order_df)


def metric_from_quantstats(portfolio_value_seq, **kwargs):
    metrics = qs.reports.metrics(portfolio_value_seq, display=False, mode="full")
    dict_ = dict(
        zip([x.replace(" ", "") for x in metrics.index], [x[0] for x in metrics.values])
    )
    return dict_


def all_metrics(
    portfolio_seq, order_df, simple_interest, verbose=True, return_dict=False
):
    ret = {}
    if not simple_interest:
        pass
        ret.update(metric_from_quantstats(portfolio_seq))
    else:
        ret.update(get_simple_interest_metrics(portfolio_seq))
    ret.update(calc_from_order_df(order_df))

    for key in list(ret.keys()):
        if str(ret[key]) == "nan":
            del ret[key]
            continue
        if ret[key] == 0:
            del ret[key]

    if verbose:
        for key, value in ret.items():
            try:
                value = np.round(value, 2)
            except:
                pass
            print(key, ":", value)

    if return_dict:
        return ret


def all_metrics_from_result(result, **kwargs):
    return all_metrics(
        result.portfolio_seq, result.order_df, result.simple_interest, **kwargs
    )


def plot_signals(data, order, n=10):
    plt.figure(figsize=(30, 10))
    order = order.copy()
    order["Date"] = order.index
    order = order[data.index[-n] : data.index[-1]]
    plt.plot(data.open[-n:])
    plt.scatter(
        order.loc[order["desired_side"] == 1, "Date"].values,
        order.loc[order["desired_side"] == 1, "order_price"].values,
        label="skitscat",
        color="green",
        s=100,
        marker="^",
    )
    plt.scatter(
        order.loc[order["desired_side"] == 0, "Date"].values,
        order.loc[order["desired_side"] == 0, "order_price"].values,
        label="skitscat",
        color="black",
        s=100,
        marker="X",
    )
    plt.scatter(
        order.loc[order["desired_side"] == -1, "Date"].values,
        order.loc[order["desired_side"] == -1, "order_price"].values,
        label="skitscat",
        color="red",
        s=100,
        marker="v",
    )


def plot_signals_from_result(result):
    plot_signals(result.portfolio_df, result.order_df, n=10)


def plot_returns(
    a,
    ax=None,
    plot_cummax=True,
    label="cumulative returns",
    color="black",
    cummax_color="red",
):
    a = (a / a[0] - 1) * 100
    if ax is None:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)

    ax.plot(a, label=label, color=color)

    if plot_cummax:
        ax.plot(a.cummax(), color=cummax_color)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns(%)")
    ax.legend()


def plot_returns_from_result(result):
    plot_returns(result.portfolio_seq)


def plot_multi_ticker_returns_from_result(result, metric_name="sharpe_ratio"):
    ax = plt.figure(figsize=(20, 10)).add_subplot(111)
    plot_returns(result.portfolio_seq, ax=ax)
    index = result.portfolio_seq.index

    for key, value in result.result_dict.items():
        value.portfolio_seq = value.portfolio_seq[index[0] :]
        value.order_df = value.order_df[index[0] :]
        value.portfolio_df = value.portfolio_df[index[0] :]
        value.metric_dict = all_metrics_from_result(
            value, verbose=False, return_dict=True
        )

    for key, value in sorted(
        result.result_dict.items(), key=lambda x: x[-1].metric_dict[metric_name]
    ):
        color = np.random.uniform(size=3)
        label = (
            key
            if metric_name is None
            else f"{key}({round(value.metric_dict[metric_name], 2)})"
        )
        plot_returns(
            value.portfolio_seq,
            ax=ax,
            label=label,
            color=color,
            cummax_color=color[::-1],
            plot_cummax=False,
        )
