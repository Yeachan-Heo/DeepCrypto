import pandas as pd
import numpy as np
import datetime
import websocket, os, sqlite3

from typing import *
from deepcrypto.backtest import *


class OrderTypes:
    MARKET = "MARKET"

    LIMIT = "LIMIT"

    MARKET_SL = "STOP_MARKET"
    LIMIT_SL = "STOP"

    MARKET_TP = "TAKE_PROFIT_MARKET"
    LIMIT_TP = "TAKE_PROFIT"


class OrderSides:
    BUY = 1
    SELL = -1


class WebsocketBrokerBase:
    def __init__(
        self,
        ticker,
        timeframe,
        strategy,
        strategy_name,
        config,
        n_bars,
        market_order=True,
        time_cut_market=True,
        stop_loss_market=True,
        take_profit_market=True,
        log_db_dir="./deepcrypto_logs",
        bet_size=None,
    ):
        self.ticker = ticker
        self.trade_asset, self.balance_asset = ticker.split("/")
        self.symbol = ticker.replace("/", "")

        self.timeframe = timeframe
        self.strategy = strategy
        self.config = config
        self.n_bars = n_bars
        self.market_order = market_order

        self.stop_loss_market = stop_loss_market
        self.take_profit_market = take_profit_market
        self.time_cut_market = time_cut_market

        self.simple_interest = False

        if not bet_size is None:
            self.simple_interest = True
            self.bet_size = bet_size

        self.url = None

        self.last_entry = np.inf

        self.data = self.init_data()

        self.index = 0

        if not os.path.exists(log_db_dir):
            os.makedirs(log_db_dir)

        log_db_path = os.path.join(log_db_dir, strategy_name + ".sqlite3")
        self.log_db = sqlite3.connect(log_db_path)

        self.log_db.execute(
            """CREATE TABLE IF NOT EXISTS PERFORMANCE (
                timestamp float,
                portfolio_value float,
                price float,
                cash_left float,
                position_size float,
                position_side int
            )"""
        )

        self.log_db.execute(
            """CREATE TABLE IF NOT EXISTS ORDERS (
                timestamp float,
                ticker text,
                type text,
                message text,
                side text,
                quantity float,
                price float
            )"""
        )

        self.log_db.execute(
            """CREATE TABLE IF NOT EXISTS ERRORS (
                timestamp float,
                error text
            )"""
        )

        self.log_db.commit()

        self.n_bars_from_last_trade = -1

    def init_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def add_to_data(self, ohlcv):
        self.data = self.data.iloc[1:].append(
            pd.DataFrame(ohlcv, index=[pd.to_datetime(ohlcv["time"])])
        )

    def preprocess_msg(self, msg):
        ohlcv = {"time": 0, "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}
        closed = False
        return ohlcv, closed

    def fetch_balance(self, asset):
        raise NotImplementedError

    def fetch_position(self):
        raise NotImplementedError
        # return position_size, position_side

    def update_account_info(self):

        cashleft = self.fetch_balance(self.balance_asset)
        position_size, position_side = self.fetch_position()

        (
            self.portfolio_value,
            self.price,
            self.cash_left,
            self.position_size,
            self.position_side,
        ) = (
            self.data.iloc[-1]["close"] * position_size + cashleft,
            self.data.iloc[-1]["close"],
            cashleft,
            position_size,
            position_side,
        )

    def order(self, quantity, type, side, price):
        raise NotImplementedError

    def cancel_all_orders(self):
        raise NotImplementedError

    def trade_step(self):

        res = self.strategy(self.data.backtest.add_defaults(), self.config).iloc[-1]

        target_pos = order_logic(
            int(self.position_side),
            bool(res["enter_long"]),
            bool(res["enter_short"]),
            res["close_long"],
            res["close_short"],
        )
        target_pos = int(target_pos)

        if target_pos != self.position_side:
            target_amount = (
                target_pos
                * res["bet"]
                * (self.portfolio_value if not self.simple_interest else self.bet_size)
                / (self.price if not self.simple_interest else 1)
            )
            order_amount = target_amount - self.position_size * self.position_side
            order_size, order_side = np.abs(order_amount), np.sign(order_amount)

            self.cancel_all_orders()
            self.order(
                order_size,
                OrderTypes.LIMIT if not self.market_order else OrderTypes.MARKET,
                order_side,
                self.price,
                message="LOGIC",
            )
            self.last_entry = np.inf if target_pos == 0 else self.index

            if not target_pos == 0:
                order_size = abs(target_amount)
                order_side = -order_side

                if str(res["stop_loss"]) != "inf":
                    order_type = (
                        OrderTypes.MARKET_SL
                        if self.stop_loss_market
                        else OrderTypes.LIMIT_SL
                    )
                    self.order(
                        order_size,
                        order_type,
                        order_side,
                        (self.price * (1 - res["stop_loss"] * target_pos)),
                        message="STOP_LOSS",
                    )

                if str(res["take_profit"]) != "inf":
                    order_type = (
                        OrderTypes.MARKET_TP
                        if self.stop_loss_market
                        else OrderTypes.LIMIT_TP
                    )
                    self.order(
                        order_size,
                        order_type,
                        order_side,
                        (self.price * (1 + res["take_profit"] * target_pos)),
                        message="TAKE_PROFIT",
                    )

        if ((self.index - self.last_entry) >= res["time_cut"]) & (
            self.position_side != 0
        ):
            self.cancel_all_orders()

            self.order(
                self.position_size,
                OrderTypes.LIMIT if not self.time_cut_market else OrderTypes.MARKET,
                -self.position_side,
                self.price,
                message="TIME_CUT",
            )

        self.index += 1

    def on_message(self, msg):
        try:
            ohlcv, closed = self.preprocess_msg(msg)

            if closed:
                self.update_account_info()
                self.add_to_data(ohlcv)
                self.log_db.execute(
                    f"""
                    INSERT INTO PERFORMANCE VALUES (
                        {datetime.datetime.now().timestamp()},
                        {self.portfolio_value},
                        {self.price},
                        {self.cash_left},
                        {self.position_size},
                        {int(self.position_side)}
                    )
                    """
                )
                self.log_db.commit()
                self.trade_step()
        except Exception as e:
            self.log_db.execute(
                f"""
                INSERT INTO ERRORS VALUES (
                    {datetime.datetime.now().timestamp()},
                    '{e}'
                )
            """
            )
            self.log_db.commit()

    def on_open(self, x):
        print("START TRADER")

    def trade(self):
        self.ws_app = websocket.WebSocketApp(
            self.url,
            on_message=lambda ws, msg: self.on_message(msg),
            on_open=lambda x: self.on_open(x),
            on_close=lambda x: print("CLOSE TRADER"),
            on_error=lambda x: print(x),
        )
        return self.ws_app.run_forever()
