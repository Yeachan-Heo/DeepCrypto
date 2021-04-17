from numba import njit
from copy import deepcopy
import numpy as np
import pandas as pd
import time


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
def get_order(bet, position_side, position_size, target_position, portfolio_value, close):
    target_amount = bet * portfolio_value / close
    order = target_amount * target_position - position_side * position_size
    order_size = np.abs(order)
    order_side = np.sign(order)
    return order_size, order_side


@njit(cache=True)
def process_order(trade_cost, slippage, cash, entry_price, position_size, position_side, order_size, order_side, close):
    close = (1 + slippage * order_side) * close


    realized, realized_percent = 0, 0

    if position_side == order_side:
        entry_price = (entry_price * position_size + order_size * close) / (position_size + order_size)
    else:
        if order_size > position_size:
            realized = (close - entry_price) * position_side * position_size
            entry_price = close
        else:
            realized = (close - entry_price) * position_side * order_size
        realized_percent  = (close / entry_price - 1) * position_side

    cash -= trade_cost * order_size * close
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
def update_portfolio_value(cash, entry_price, close, position_side, position_size):

    ret = cash + (close - entry_price) * position_side * position_size
    #print(cash, ret, entry_price, close, position_side, position_size)
    return ret

@njit(cache=True)
def run_backtest_compiled(
        initial_cash,
        timestamp_seq,
        close_seq,
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
    close = 0

    portfolio_value = update_portfolio_value(cash, entry_price, close, position_side, position_size)

    last_entry = np.inf
    cnt = 0
    for i, timestamp in enumerate(timestamp_seq):
        timestamp, \
        close, \
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
            close_seq[i],
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
        price=close

        unrealized_pnl_percent = (close/entry_price - 1) * position_side if position_side else 0

        if position_side:
            take_profit_flag, stop_loss_flag = 0, 0
            time_cut_flag = (timestamp - last_entry) > time_cut
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
                # print("close", cnt, time_cut_flag, stop_loss_flag, take_profit_flag)

        target_position = order_logic(
            position_side, enter_long, enter_short, close_long, close_short)

        order_size, order_side = get_order(bet, position_side, position_size, target_position, portfolio_value, price)
        #print(position_side, position_size)
        if position_side != target_position:
            temp = position_side

            cnt += 1

            realized, realized_percent, cash, entry_price, position_size, position_side = process_order(
                trade_cost, slippage, cash, entry_price, position_size, position_side, order_size, order_side, price)

            if realized:
                order_logger.append((timestamp, realized, realized_percent, temp))
                #print(order_logger[-1])

            if (((not temp) and position_side) or (temp and (temp != position_side))):
                last_entry = timestamp
            elif not position_side:
                last_entry = np.inf

        portfolio_value = update_portfolio_value(cash, entry_price, close, position_side, position_size)
        portfolio_value_logger.append((timestamp, portfolio_value, cash, close, entry_price, position_side, position_size, unrealized_pnl_percent))


    return order_logger, portfolio_value_logger


def make_constant_seq(value, length):
    return np.ones(length) * value


def check_variable(var, length, name):
    if not isinstance(var, np.ndarray):
        if isinstance(var, type(None)):
            raise ValueError(f"{name} is None")
        var = make_constant_seq(var, length)
    return var

def run_backtest(
        initial_cash=10000,
        timestamp: pd.Series=None,
        close: pd.Series=None,
        high: pd.Series=None,
        low: pd.Series=None,
        enter_long: pd.Series=None,
        enter_short: pd.Series=None,
        close_long: pd.Series=None,
        close_short: pd.Series=None,
        bet=1,
        trade_cost=0,
        slippage=0,
        time_cut=np.inf,
        stop_loss=np.inf,
        take_profit=np.inf,
        low_first=1,
        log_time=True,
        **kwargs
):
    if isinstance(enter_long, (pd.Series, np.ndarray)):
        enter_long = pd.Series(enter_long).shift(1)
    if isinstance(enter_short, (pd.Series, np.ndarray)):
        enter_short = pd.Series(enter_short).shift(1)
    if isinstance(close_long, (pd.Series, np.ndarray)):
        close_long = pd.Series(close_long).shift(1)
    if isinstance(close_short, (pd.Series, np.ndarray)):
        close_short = pd.Series(close_short).shift(1)
    if isinstance(bet, (pd.Series, np.ndarray)):
        bet = pd.Series(bet).shift(1)
    if isinstance(stop_loss, pd.Series) or np.array(stop_loss).shape != ():
        stop_loss = pd.Series(stop_loss).shift(1)
    if isinstance(take_profit, pd.Series) or np.array(take_profit).shape != ():
        take_profit = pd.Series(take_profit).shift(1)
    if isinstance(time_cut, pd.Series) or np.array(time_cut).shape != ():
        time_cut = pd.Series(time_cut).shift(1)

    s = time.time()

    if timestamp is None or close is None:
        raise ValueError("must specify timestamp and close")

    if (not stop_loss == np.inf or not take_profit == np.inf) and (high is None or low is None):
        raise ValueError("must specify timestamp and close")

    length = timestamp.shape[0]

    if isinstance(enter_long, type(None)):
        enter_long = make_constant_seq(0, length).astype(bool)
    if isinstance(enter_short, type(None)):
        enter_short = make_constant_seq(0, length).astype(bool)
    if isinstance(close_long, type(None)):
        close_long = make_constant_seq(0, length).astype(bool)
    if isinstance(close_short, type(None)):
        close_short = make_constant_seq(0, length).astype(bool)
    if isinstance(high, type(None)):
        high = make_constant_seq(0, length)
    if isinstance(low, type(None)):
        low = make_constant_seq(0, length)

    bet = check_variable(bet, length, "bet")
    trade_cost = check_variable(trade_cost, length, "trade_cost")
    slippage = check_variable(slippage, length, "slippage")
    time_cut = check_variable(time_cut, length, "time_cut")
    stop_loss = check_variable(stop_loss, length, "stop_loss")
    take_profit = check_variable(take_profit, length, "take_profit")
    low_first = check_variable(low_first, length, "low_first")


    order_logger, portfolio_logger = run_backtest_compiled(
        np.array(initial_cash).astype(np.float64),
        np.array(timestamp).astype(np.int64),
        np.array(close).astype(np.float64),
        np.array(high).astype(np.float64),
        np.array(low).astype(np.float64),
        np.array(enter_long).astype(np.bool),
        np.array(enter_short).astype(np.bool),
        np.array(close_long).astype(np.bool),
        np.array(close_short).astype(np.bool),
        np.array(bet).astype(np.float64),
        np.array(trade_cost).astype(np.float64),
        np.array(slippage).astype(np.float64),
        np.array(time_cut).astype(np.float64),
        np.array(stop_loss).astype(np.float64),
        np.array(take_profit).astype(np.float64),
        np.array(low_first).astype(np.float64)
    )

    order_df = pd.DataFrame(order_logger, columns=["timestamp", "realized", "realized_percent", "side"])
    portfolio_df = pd.DataFrame(portfolio_logger, columns=["timestamp", "portfolio_value", "cash", "close", "entry_price", "position_side", "position_size", "unrealized_pnl_percent"])
    portfolio_df.index = pd.to_datetime(portfolio_df["timestamp"])

    if log_time:
        print(f"backtest complete in {time.time() - s} seconds")

    return order_df, portfolio_df.drop_duplicates("timestamp")


@pd.api.extensions.register_dataframe_accessor("lightning_backtest")
class BacktestAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def add_defaults(self):
        self._obj["enter_long"] = 0
        self._obj["enter_short"] = 0
        self._obj["close_long"] = 0
        self._obj["close_short"] = 0

        self._obj["bet"] = 1

        self._obj["stop_loss"] = np.inf
        self._obj["take_profit"] = np.inf
        self._obj["time_cut"] = np.inf

        self._obj["trade_cost"] = 0
        self._obj["slippage"] = 0

        if not "low_first" in self._obj.columns:
            self._obj["low_first"] = 1

        return self._obj

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
            df["close"].values.astype(np.float64),
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

        order_df = pd.DataFrame(order_logger, columns=["timestamp", "realized", "realized_percent", "side"])
        portfolio_df = pd.DataFrame(portfolio_logger,
                                    columns=["timestamp", "portfolio_value", "cash", "close", "entry_price",
                                             "position_side", "position_size", "unrealized_pnl_percent"])
        portfolio_df.index = pd.to_datetime(portfolio_df["timestamp"])

        if log_time:
            print(f"backtest completed in {time.time() - t} seconds")

        return portfolio_df, order_df

def strategy(fn):
    def wrapped(df):
        df = fn(df)
        return df.lightning_backtest().run()


def test_ma_crossover():
    from deepcrypto.data_utils.crawlers.bitmex import load_bitmex_data


    data = load_bitmex_data("/home/ych/Storage/bitmex/bitmex.db", "1H", "XBTUSD")

    data = data.lightning_backtest.add_defaults()

    data["fastma"] = data["close"].rolling(15).mean()
    data["slowma"] = data["close"].rolling(150).mean()

    data["vol_diff"] = data["volume"] / data["volume"].rolling(50).mean() > 2

    data["enter_long"] = data["fastma"] > data["slowma"]
    data["enter_short"] = data["slowma"] > data["fastma"]

    data["bet"] = 1

    data["trade_cost"] = 0.0005
    data["take_profit"] = 0.2
    data["stop_loss"] = 0.1

    data["time_cut"] = DAY * 7

    data, order_df = data.lightning_backtest.run()
    order_df.to_csv("./order.csv")

    import quantstats as qs

    qs.reports.html(data["portfolio_value"].resample("1D").last(), benchmark=data["close"].resample("1D").last(), output="./out.html")

if __name__ == '__main__':
    test_ma_crossover()

