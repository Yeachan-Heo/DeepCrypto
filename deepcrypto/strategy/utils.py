import pandas as pd 

def breakout(seq: pd.Series):
    return (seq == 1) & (seq.diff() == 1)

def downsample_df(df, freq):
    df = df[["open", "high", "low", "close", "volume"]]
    
    df = df.resample(freq).agg({
        "open" : "first",
        "high" : "max",
        "low" : "min",
        "close" : "last",
        "volume" : "sum"
    })

    return df.shift(1)

def downsample_seq(seq, freq, by):
    return seq.resample(freq).apply(by)

def upsample_seq(df, freq):
    return df.resample(freq).last()

def ma_perc(seq, freq):
    return seq/seq.rolling(freq).mean()