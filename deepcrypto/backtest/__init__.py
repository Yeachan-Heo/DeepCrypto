from numba import njit
from copy import deepcopy
import numpy as np
import pandas as pd
import time
import ray


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
def get_order(
    bet, position_side, position_size, target_position, portfolio_value, open
):
    target_amount = bet * portfolio_value / open
    order = target_amount * target_position - position_side * position_size
    order_size = np.abs(order)
    order_side = np.sign(order)
    return order_size, order_side


@njit(cache=True)
def process_order(
    trade_cost,
    slippage,
    cash,
    entry_price,
    position_size,
    position_side,
    order_size,
    order_side,
    open,
):
    open = (1 + slippage * order_side) * open

    realized, realized_percent = 0.0, 0.0

    if position_side == order_side:
        entry_price = (entry_price * position_size + order_size * open) / (
            position_size + order_size
        )
    else:
        if order_size > position_size:
            realized = (open - entry_price) * position_side * position_size
            entry_price = open
        else:
            realized = (open - entry_price) * position_side * order_size
        realized_percent = (open / entry_price - 1) * position_side

    fee = trade_cost * order_size * open
    cash -= fee
    cash += realized
    position = position_side * position_size + order_size * order_side

    position_side, position_size = np.sign(position), np.abs(position)

    if not position_side:
        entry_price = 0

    return (
        realized,
        realized_percent,
        cash,
        entry_price,
        position_size,
        position_side,
        fee,
        open,
    )


@njit(cache=True)
def update_portfolio_value(cash, entry_price, open, position_side, position_size):
    ret = cash + (open - entry_price) * position_side * position_size
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
    simple_interest=False,
):

    portfolio_value_logger = []
    order_logger = []

    position_side = 0
    position_size = 0

    entry_price = 0

    cash = initial_cash
    open = 0

    portfolio_value = update_portfolio_value(
        cash, entry_price, open, position_side, position_size
    )

    last_entry = np.inf

    for i, timestamp in enumerate(timestamp_seq):
        (
            timestamp,
            open,
            high,
            low,
            enter_long,
            enter_short,
            close_long,
            close_short,
            bet,
            trade_cost,
            slippage,
            time_cut,
            stop_loss,
            take_profit,
        ) = (
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
        )
        price = open

        target_position = order_logic(
            position_side, enter_long, enter_short, close_long, close_short
        )

        max_position_size = portfolio_value if not simple_interest else initial_cash
        order_size, order_side = get_order(
            bet, position_side, position_size, target_position, max_position_size, price
        )

        if position_side != target_position:

            temp = position_side

            (
                realized,
                realized_percent,
                cash,
                entry_price,
                position_size,
                position_side,
                fee,
                price,
            ) = process_order(
                trade_cost,
                slippage,
                cash,
                entry_price,
                position_size,
                position_side,
                order_size,
                order_side,
                price,
            )

            hold_bars = i - last_entry if realized else 0

            order_logger.append(
                (
                    timestamp,
                    realized,
                    fee,
                    "LC",
                    realized_percent,
                    temp,
                    target_position,
                    hold_bars,
                    price,
                )
            )

            if ((not temp) and position_side) or (temp and (temp != position_side)):
                last_entry = i
                sl_val, tp_val = stop_loss, take_profit

            elif not position_side:
                last_entry = np.inf

        unrealized_pnl_percent = (
            (open / entry_price - 1) * position_side if position_side else 0
        )

        if position_size:

            take_profit_flag, stop_loss_flag = 0, 0
            time_cut_flag = (i - last_entry) >= time_cut

            if not time_cut_flag:
                unrealized_pnl_percent = (
                    (low if position_side == 1 else high) / entry_price - 1
                ) * position_side
                stop_loss_flag = unrealized_pnl_percent <= -sl_val

                if stop_loss_flag:
                    price = entry_price * (1 + position_side * -sl_val)

                if not stop_loss_flag:
                    unrealized_pnl_percent = (
                        (high if position_side == 1 else low) / entry_price - 1
                    ) * position_side
                    take_profit_flag = unrealized_pnl_percent >= tp_val

                    if take_profit_flag:
                        price = entry_price * (1 + position_side * tp_val)

            if time_cut_flag or stop_loss_flag or take_profit_flag:
                if position_side == 1:
                    close_long = 1
                elif position_side == -1:
                    close_short = 1

        target_position = order_logic(
            position_side, enter_long, enter_short, close_long, close_short
        )

        order_size, order_side = get_order(
            bet, position_side, position_size, target_position, portfolio_value, price
        )

        if position_side != target_position:
            temp = position_side

            (
                realized,
                realized_percent,
                cash,
                entry_price,
                position_size,
                position_side,
                fee,
                price,
            ) = process_order(
                trade_cost,
                slippage,
                cash,
                entry_price,
                position_size,
                position_side,
                order_size,
                order_side,
                price,
            )

            hold_bars = i - last_entry if realized else 0

            reason = "TC" if time_cut_flag else "SL" if stop_loss_flag else "TP"

            order_logger.append(
                (
                    timestamp,
                    realized,
                    fee,
                    reason,
                    realized_percent,
                    temp,
                    target_position,
                    hold_bars,
                    price,
                )
            )

            if ((not temp) and position_side) or (temp and (temp != position_side)):
                last_entry = i
                sl_val, tp_val = stop_loss, take_profit

            elif not position_side:
                last_entry = np.inf

        portfolio_value = update_portfolio_value(
            cash, entry_price, open, position_side, position_size
        )

        portfolio_value_logger.append(
            (
                timestamp,
                portfolio_value,
                cash,
                open,
                entry_price,
                position_side,
                position_size,
                unrealized_pnl_percent,
            )
        )

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

        return df

    def run(self, initial_cash=10000, simple_interest=False, log_time=True):
        df = deepcopy(self._obj)
        return run_backtest_df(df, initial_cash, simple_interest, log_time)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class BacktestResults:
    def __init__(self, order_df, portfolio_df, simple_interest):
        self.order_df = order_df
        self.portfolio_df = portfolio_df
        self.simple_interest = simple_interest
        self.portfolio_seq = self.portfolio_df.portfolio_value.resample("1D").last()


def run_backtest_df(df, initial_cash=10000, simple_interest=False, log_time=True):

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
        np.float64(initial_cash),
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
        simple_interest,
    )

    order_df = pd.DataFrame(
        order_logger,
        columns=[
            "timestamp",
            "realized",
            "fee",
            "reason",
            "realized_percent",
            "prev_side",
            "desired_side",
            "hold_bars",
            "order_price",
        ],
    )
    portfolio_df = pd.DataFrame(
        portfolio_logger,
        columns=[
            "timestamp",
            "portfolio_value",
            "cash",
            "open",
            "entry_price",
            "position_side",
            "position_size",
            "unrealized_pnl_percent",
        ],
    )
    portfolio_df.index = pd.to_datetime(portfolio_df["timestamp"])
    order_df.index = pd.to_datetime(order_df["timestamp"])

    if log_time:
        print(f"backtest completed in {time.time() - t} seconds")

    return BacktestResults(order_df, portfolio_df, simple_interest)


class MultiTickerBacktestResults:
    def __init__(self, portfolio_seq, order_df, result_dict, simple_interest):
        self.portfolio_seq = portfolio_seq.resample("1D").last()
        self.order_df = order_df
        self.result_dict = result_dict
        self.simple_interest = simple_interest


@ray.remote
def run_backtest_ray(strategy, config, df, **kwargs):
    result_df = run_backtest_df(strategy(df, config), **kwargs)
    return result_df


def do_multi_ticker_backtest(
    strategy, data_dict, config, log_time=False, simple_interest=False
):
    result_dict = {
        ticker: run_backtest_ray.remote(
            strategy, config, df, log_time=log_time, simple_interest=simple_interest
        )
        for ticker, df in data_dict.items()
    }

    result_dict = {ticker: ray.get(df) for ticker, df in result_dict.items()}

    order_df_lst = [v.order_df for v in result_dict.values()]

    for ticker, order_df in zip(result_dict.keys(), order_df_lst):
        order_df["ticker"] = ticker

    portfolio_df_lst = [v.portfolio_df for v in result_dict.values()]

    index_len = np.inf
    index = None

    for port_df in portfolio_df_lst:
        if len(port_df.index) < index_len:
            index_len = len(port_df.index)
            index = port_df.index

    portfolio_seq = portfolio_df_lst[0].portfolio_value.reindex(index)
    portfolio_seq /= portfolio_seq.iloc[0]

    for port_df in portfolio_df_lst[1:]:
        pfseq = port_df.portfolio_value.reindex(index)
        portfolio_seq += pfseq / pfseq.iloc[0]

    return MultiTickerBacktestResults(
        portfolio_seq,
        pd.concat(order_df_lst).sort_index(),
        result_dict,
        simple_interest,
    )


def get_longest_index(data_dict):
    longest_index = []
    for df in data_dict.values():
        if len(longest_index) < df.index:
            longest_index = df.index
    return df.index


def calc_avaliable_tickers(data_dict, from_date, to_date):
    tickers = []
    for ticker, df in data_dict.items():
        try:
            sliced = df[from_date:to_date]
        except IndexError:
            continue
        if (sliced.index[0] != from_date) | (sliced.index[-1] == to_date):
            continue
        tickers.append(ticker)
    return tickers


def do_rotational_multi_ticker_backtest(
    strategy, ticker_rotation_strategy, data_dict, period="M"
):
    longest_index = get_longest_index(data_dict)
