def breakout(seq):
    seq = seq.astype(bool)
    return (seq.shift(1) == False) & seq
