import pandas as pd
import numpy as np
import json
import ccxt

from deepcrypto.brokers.ws_broker import *


class BinanceFuturesBroker(WebsocketBrokerBase):
    def __init__(
        self,
        api_key,
        api_secret,
        leverage=2,
        contract_type="perpetual",
        bet_size=None,
        **kwargs,
    ):
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                },
            }
        )

        if not contract_type in ["perpetual", "current_quarter", "next_quarter"]:
            raise ValueError("invalid contract type")

        super(BinanceFuturesBroker, self).__init__(**kwargs, bet_size=bet_size)

        self.url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}_{contract_type}@continuousKline_{self.timeframe}"

        self.trade_asset, self.balance_asset = self.ticker.split("/")
        self.n_bars_from_last_order = 0

        self.leverage = leverage

    def init_data(self) -> pd.DataFrame:
        data = self.exchange.fetch_ohlcv(
            symbol=self.ticker, timeframe=self.timeframe, limit=self.n_bars
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
        ohlcv = {
            "time": msg["k"]["t"] * 1000000,
            "open": float(msg["k"]["o"]),
            "high": float(msg["k"]["h"]),
            "low": float(msg["k"]["l"]),
            "close": float(msg["k"]["c"]),
            "volume": float(msg["k"]["v"]),
        }
        closed = bool(msg["k"]["x"])
        return ohlcv, closed

    def update_account_info(self):
        account_info = self.exchange.fetch_balance()

        cashleft = account_info["total"][self.balance_asset]
        position = list(
            filter(
                lambda x: x["symbol"] == self.symbol, account_info["info"]["positions"]
            )
        )[0]
        unrealized = float(position["unrealizedProfit"])
        position_size = float(position["positionAmt"])

        position_side = int(np.sign(position_size))

        price = self.exchange.fetch_ohlcv(
            symbol=self.ticker, timeframe=self.timeframe, limit=1
        )[-1][0]

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
        self.exchange.fapiPrivate_post_leverage(
            {"symbol": self.symbol, "leverage": self.leverage}
        )

        params = {}

        if not order_type in [OrderTypes.LIMIT, OrderTypes.MARKET]:
            params = {"stopPrice": price}

        try:
            self.exchange.create_order(
                symbol=self.ticker,
                side="SELL" if int(side) < 0 else "BUY",
                type=order_type,
                params=params,
                amount=quantity,
                price=None if self.market_order else price,
                timeInForce="GTC",
                stopPrice=price,
                recvWindow=5000,
            )

            self.log_db.execute(
                f"""
                INSERT INTO ORDERS VALUES (
                    {datetime.datetime.now().timestamp()},
                    '{self.ticker}'
                    '{order_type}',
                    '{message}',
                    '{"SELL" if int(side) < 0 else "BUY"}',
                    {quantity},
                    {price}
                )
                """
            )
            self.log_db.commit()

        except Exception as exception:
            print("order error: ", exception)

    def cancel_all_orders(self):
        orders = self.exchange.fetch_open_orders(symbol=self.ticker)
        for order in orders:
            self.exchange.cancel_order(symbol=self.ticker, id=order["info"]["orderId"])
