import pandas as pd

def make_periodic_indicator_fn(indicator_fn, freq="1D"):
    def wrapped(inp):
        x = inp.resample(freq).aggregate({"open" : "first", "high" : "max", "low" : "min", "close" : "last", "volume" : "sum"})
        indicator = pd.Series(indicator_fn(x), index=x.index).shift(1)[inp.index.floor(freq)]
        indicator.index = inp.index
        return indicator.values
    return wrapped

def make_periodic_indicator(indicator_fn, freq, df):
    return make_periodic_indicator_fn(indicator_fn, freq)(df)

def add_new_timeframe(df, freq):
    df_ = df[["open", "high", "low", "close"]]
    open_ = make_periodic_indicator_fn(lambda x: x["open"])(df_)
    high_ = make_periodic_indicator_fn(lambda x: x["high"])(df_)
    low_ = make_periodic_indicator_fn(lambda x: x["low"])(df_)
    close_ = make_periodic_indicator_fn(lambda x: x["close"])(df_)
    volume_ = make_periodic_indicator_fn(lambda x: x["volume"])(df_)

    df[f"open_{freq}"] = open_
    df[f"high_{freq}"] = high_
    df[f"low_{freq}"] = low_
    df[f"close_{freq}"] = close_
    df[f"volume_{freq}"] = volume_

    return df