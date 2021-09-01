import pandas as pd
import numpy as np
import json
import ccxt

from deepcrypto.brokers.ws_broker import *
import bybit


class BybitBroker(WebsocketBrokerBase):
    def __init__(
        self,
        api_key,
        api_secret,
        leverage=2,
        bet_size=None,
        post_only=True,
        **kwargs,
    ):
        self.exchange = bybit.bybit(api_key=api_key, api_secret=api_secret, test=False)

        self.ccxt_exchange = ccxt.bybit(
            {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
        )

        super(BybitBroker, self).__init__(**kwargs, bet_size=bet_size)

        self.url = "wss://stream.bybit.com/realtime"

        self.trade_asset, self.balance_asset = self.ticker.split("/")
        self.n_bars_from_last_order = 0

        self.leverage = leverage
        self.post_only = post_only

        self.asset, self.base_currency = self.ticker.split("/")

    def init_data(self) -> pd.DataFrame:
        data = self.ccxt_exchange.fetch_ohlcv(
            self.ticker, timeframe=self.timeframe + "m", limit=200
        )

        data = data[:-1]

        data = pd.DataFrame(
            data, columns=["time", "open", "high", "low", "close", "volume"]
        )

        data["time"] = data["time"] * 1000000
        data.index = pd.to_datetime(data["time"])
        return data

    def preprocess_msg(self, msg):
        msg = json.loads(msg)

        if not "data" in msg.keys():
            return [], True

        ohlcv = {
            "time": msg["data"][0]["timestamp"],
            "open": float(msg["data"][0]["open"]),
            "high": float(msg["data"][0]["high"]),
            "low": float(msg["data"][0]["low"]),
            "close": float(msg["data"][0]["close"]),
            "volume": float(msg["data"][0]["volume"]),
        }

        closed = bool(msg["data"][0]["confirm"])

        return ohlcv, closed

    def update_account_info(self):
        cashleft = self.exchange.Wallet.Wallet_getBalance(coin=self.asset).result()
        cashleft = float(cashleft[0]["result"][self.asset]["equity"])

        position = self.exchange.Positions.Positions_myPosition(
            symbol=self.symbol
        ).result()
        position = position[0]["result"]

        unrealized = float(position["unrealised_pnl"])
        position_size = float(position["size"])
        position_side = 1 if position["side"] == "Buy" else -1

        if position_size is 0:
            position_side = 0

        price = self.ccxt_exchange.fetch_ohlcv(
            self.ticker, timeframe=self.timeframe + "m", limit=1
        )
        price = price[0][1]

        (
            self.portfolio_value,
            self.price,
            self.cash_left,
            self.position_size,
            self.position_side,
        ) = (
            cashleft + unrealized,
            price,
            cashleft,
            abs(position_size),
            position_side,
        )

    def order(self, quantity, order_type, side, price, message, **kwargs):

        params = {}

        order_type = {OrderTypes.LIMIT: "Limit", OrderTypes.MARKET: "Market"}[
            order_type
        ]

        try:
            res = self.exchange.Order.Order_new(
                order_type=order_type,
                side=("Buy" if side == OrderSides.BUY else "Sell"),
                symbol=self.symbol,
                qty=str(int(quantity)),
                price=str(price),
                time_in_force=("PostOnly" if self.post_only else "GoodTillCancel"),
            ).result()

        except Exception as exception:
            print("order error: ", exception)

    def on_open(self, ws):
        print("START TRADER")
        try:
            msg = '{"op" : "subscribe", "args":["klineV2.%s.%s"]}' % (
                self.timeframe,
                self.symbol,
            )
        except Exception as e:
            print(e)
        print(msg)
        ws.send(msg)

    def cancel_all_orders(self):
        self.exchange.Order.Order_cancelAll(symbol=self.symbol).result()
