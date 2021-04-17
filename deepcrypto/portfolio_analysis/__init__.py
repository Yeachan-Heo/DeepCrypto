import quantstats as qs

def _calc_from_order_df(order_df, label="bothside"):
    a = {}
    a[f"{label}_win_rate"] = order_df.realized[order_df.realized >= 0].count()/order_df.realized.count() * 100
    a[f"{label}_profit_factor"] = -order_df.realized[order_df.realized >= 0].mean()/order_df.realized[order_df.realized < 0].mean()
    a[f"{label}_total_profit"] = order_df.realized.sum()
    return a

def calc_from_order_df(order_df):
    s_df = order_df[order_df.side < 0]
    b_df = order_df[order_df.side > 0]
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
