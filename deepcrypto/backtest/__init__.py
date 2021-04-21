from numba import njit
from copy import deepcopy
import numpy as np
import pandas as pd
import time
import os

class TIMEFRAMES:
    MINUTE = 86400000000000 / 24 / 60
    HOUR = 86400000000000 / 24
    DAY = HOUR * 24

@njit(cache=True)
def order_logic(position_side, enter_long, enter_short, close_long, close_short):
    if position_side == 0:
        if enter_long:
            if enter_short:
                return 0
            if close_long:
                return 0
            return 1

        if enter_short:
            if enter_long:
                return 0
            if close_short:
                return 0
            return -1
        return 0

    elif position_side == 1:
        if enter_short:
            return -1
        if close_long:
            return 0
        return 1

    elif position_side == -1:
        if enter_long:
            return 1
        if close_short:
            return 0
        return -1

@njit(cache=True)
def get_order(bet, position_side, position_size, target_position, portfolio_value, open):
    target_amount = bet * portfolio_value / open
    order = target_amount * target_position - position_side * position_size
    order_size = np.abs(order)
    order_side = np.sign(order)
    return order_size, order_side


@njit(cache=True)
def process_order(trade_cost, slippage, cash, entry_price, position_size, position_side, order_size, order_side, open):
    open = (1 + slippage * order_side) * open


    realized, realized_percent = 0., 0.

    if position_side == order_side:
        entry_price = (entry_price * position_size + order_size * open) / (position_size + order_size)
    else:
        if order_size > position_size:
            realized = (open - entry_price) * position_side * position_size
            entry_price = open
        else:
            realized = (open - entry_price) * position_side * order_size
        realized_percent  = (open / entry_price - 1) * position_side

    cash -= trade_cost * order_size * open
    cash += realized
    position = position_side * position_size + order_size * order_side

    position_side, position_size = np.sign(position), np.abs(position)

    if not position_side:
        entry_price = 0


    #if realized:
        #print(f"realized : {realized}")
        #print(cash)

        #print("realized :{} \nrealized_percent: {}\n cash: {}\n entry_price:{} \nposition_size:{}\n position_side:{}\n\n".format(realized, realized_percent, cash, entry_price, position_size, position_side))

    return realized, realized_percent, cash, entry_price, position_size, position_side

## get_order, process_order 수

@njit(cache=True)
def update_portfolio_value(cash, entry_price, open, position_side, position_size):

    ret = cash + (open - entry_price) * position_side * position_size
    #print(cash, ret, entry_price, open, position_side, position_size)
    return ret

@njit(cache=True)
def run_backtest_compiled(
        initial_cash,
        timestamp_seq,
        open_seq,
        high_seq,
        low_seq,
        enter_long_seq,
        enter_short_seq,
        close_long_seq,
        close_short_seq,
        bet_seq,
        trade_cost_seq,
        slippage_seq,
        time_cut_seq,
        stop_loss_seq,
        take_profit_seq,
        low_first_seq):

    portfolio_value_logger = []
    order_logger = []

    position_side = 0
    position_size = 0

    entry_price = 0

    cash = initial_cash
    open = 0

    portfolio_value = update_portfolio_value(cash, entry_price, open, position_side, position_size)

    last_entry = np.inf
    cnt = 0
    for i, timestamp in enumerate(timestamp_seq):
        timestamp, \
        open, \
        high, \
        low, \
        enter_long, \
        enter_short, \
        close_long, \
        close_short, \
        bet, \
        trade_cost, \
        slippage, \
        time_cut, \
        stop_loss, \
        take_profit, \
        low_first = (
            timestamp_seq[i],
            open_seq[i],
            high_seq[i],
            low_seq[i],
            enter_long_seq[i],
            enter_short_seq[i],
            close_long_seq[i],
            close_short_seq[i],
            bet_seq[i],
            trade_cost_seq[i],
            slippage_seq[i],
            time_cut_seq[i],
            stop_loss_seq[i],
            take_profit_seq[i],
            low_first_seq[i]
        )
        price=open

        target_position = order_logic(
            position_side, enter_long, enter_short, close_long, close_short)

        order_size, order_side = get_order(bet, position_side, position_size, target_position, portfolio_value, price)

        if position_side != target_position:
            temp = position_side

            cnt += 1

            realized, realized_percent, cash, entry_price, position_size, position_side = process_order(
                trade_cost, slippage, cash, entry_price, position_size, position_side, order_size, order_side, price)

            hold_bars = i - last_entry if realized else 0

            order_logger.append((timestamp, realized, realized_percent, temp, target_position, hold_bars, price))
                #print(order_logger[-1])

            if (((not temp) and position_side) or (temp and (temp != position_side))):
                last_entry = i

            elif not position_side:
                last_entry = np.inf


        unrealized_pnl_percent = (open/entry_price - 1) * position_side if position_side else 0

        if position_side:
            take_profit_flag, stop_loss_flag = 0, 0
            time_cut_flag = (i - last_entry) >= time_cut
            # print(timestamp, last_entry, timestamp - last_entry, time_cut)

            if low_first:
                unrealized_pnl_percent = (low/entry_price - 1) * position_side
                take_profit_flag = unrealized_pnl_percent >= take_profit
                stop_loss_flag = unrealized_pnl_percent <= -stop_loss
                if stop_loss_flag or take_profit_flag:
                    price=entry_price * (1 + position_side * (take_profit if take_profit_flag else -stop_loss))
            else:
                if (not take_profit_flag) and (not stop_loss_flag):
                    unrealized_pnl_percent = (high / entry_price - 1) * position_side
                    take_profit_flag = unrealized_pnl_percent >= take_profit
                    stop_loss_flag = unrealized_pnl_percent <= -stop_loss
                    if stop_loss_flag or take_profit_flag:
                        price=entry_price * (1 + position_side * (take_profit if take_profit_flag else -stop_loss))

            if (time_cut_flag or stop_loss_flag or take_profit_flag):
                if position_side == 1:
                    close_long = 1
                elif position_side == -1:
                    close_short = 1
                cnt += 1
                # print("open", cnt, time_cut_flag, stop_loss_flag, take_profit_flag)

        target_position = order_logic(
            position_side, enter_long, enter_short, close_long, close_short)

        order_size, order_side = get_order(bet, position_side, position_size, target_position, portfolio_value, price)

        if position_side != target_position:
            temp = position_side

            cnt += 1

            realized, realized_percent, cash, entry_price, position_size, position_side = process_order(
                trade_cost, slippage, cash, entry_price, position_size, position_side, order_size, order_side, price)

            hold_bars = i - last_entry if realized else 0

            order_logger.append((timestamp, realized, realized_percent, temp, target_position, hold_bars, price))
                #print(order_logger[-1])

            if (((not temp) and position_side) or (temp and (temp != position_side))):
                last_entry = timestamp
            elif not position_side:
                last_entry = np.inf

        portfolio_value = update_portfolio_value(cash, entry_price, open, position_side, position_size)
        portfolio_value_logger.append((timestamp, portfolio_value, cash, open, entry_price, position_side, position_size, unrealized_pnl_percent))


    return order_logger, portfolio_value_logger


@pd.api.extensions.register_dataframe_accessor("backtest")
class BacktestAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def add_defaults(self):
        df = deepcopy(self._obj)
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["close_long"] = 0
        df["close_short"] = 0

        df["bet"] = 1

        df["stop_loss"] = np.inf
        df["take_profit"] = np.inf
        df["time_cut"] = np.inf

        df["trade_cost"] = 0
        df["slippage"] = 0

        if not "low_first" in df.columns:
            df["low_first"] = 1

        return df

    def run(self, initial_cash=10000, log_time=True):
        df = deepcopy(self._obj)

        df["enter_long"] = df["enter_long"].shift(1)
        df["enter_short"] = df["enter_short"].shift(1)
        df["close_long"] = df["close_long"].shift(1)
        df["close_short"] = df["close_short"].shift(1)
        df["bet"] = df["bet"].shift(1)
        df["stop_loss"] = df["stop_loss"].shift(1)
        df["take_profit"] = df["take_profit"].shift(1)
        df["time_cut"] = df["time_cut"].shift(1)

        df = df.ffill().dropna()


        t = time.time()

        order_logger, portfolio_logger = run_backtest_compiled(
            np.array(initial_cash).astype(np.float64),
            df.index.values.astype(np.int64),
            df["open"].values.astype(np.float64),
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["enter_long"].values.astype(np.float64),
            df["enter_short"].values.astype(np.float64),
            df["close_long"].values.astype(np.float64),
            df["close_short"].values.astype(np.float64),
            df["bet"].values.astype(np.float64),
            df["trade_cost"].values.astype(np.float64),
            df["slippage"].values.astype(np.float64),
            df["time_cut"].values.astype(np.float64),
            df["stop_loss"].values.astype(np.float64),
            df["take_profit"].values.astype(np.float64),
            df["low_first"].values.astype(bool)
        )

        order_df = pd.DataFrame(order_logger, columns=["timestamp", "realized", "realized_percent", "prev_side", "desired_side", "hold_bars", "order_price"])
        portfolio_df = pd.DataFrame(portfolio_logger,
                                    columns=["timestamp", "portfolio_value", "cash", "open", "entry_price",
                                             "position_side", "position_size", "unrealized_pnl_percent"])
        portfolio_df.index = pd.to_datetime(portfolio_df["timestamp"])

        if log_time:
            print(f"backtest completed in {time.time() - t} seconds")

        return portfolio_df, order_df

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

def strategy(fn):
    def wrapped(config, df):
        df.backtest.add_defaults()
        df = fn(config, df)
        return df
    return wrapped


def test_ma_crossover():
    from deepcrypto.data_utils.crawlers.bitmex import load_bitmex_data


    data = load_bitmex_data("/home/ych/Storage/bitmex/bitmex.db", "1H", "XBTUSD")

    data = data.backtest.add_defaults()

    data["fastma"] = data["open"].rolling(15).mean()
    data["slowma"] = data["open"].rolling(150).mean()

    data["vol_diff"] = data["volume"] / data["volume"].rolling(50).mean() > 5

    data["enter_long"] = data["fastma"] > data["slowma"]
    data["enter_short"] = data["slowma"] > data["fastma"]

    data["bet"] = 1

    data["trade_cost"] = 0.001
    data["take_profit"] = 0.002
    data["stop_loss"] = 0.001

    data["time_cut"] = TIMEFRAMES.DAY * 7

    data, order_df = data.backtest.run()
    order_df.to_csv("./order.csv")

    import quantstats as qs

    qs.reports.html(data["portfolio_value"].resample("1D").last(), benchmark=data["open"].resample("1D").last(), output="./out.html")

if __name__ == '__main__':
    test_ma_crossover()

