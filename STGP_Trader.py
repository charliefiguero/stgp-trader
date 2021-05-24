""" GP_Entity will initialise multiple of these and give them each an improvement function dictated by the tree of the individual."""

import sys
import random
from time import sleep
from typing import List
import json

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader

class Order_Data():
    """ used for logging """
    def __init__(self, customer_price, trans_price, posted_improve, actual_improve, exchange_msg):
        self.customer_price = customer_price
        self.trans_price = trans_price
        self.posted_improve = posted_improve
        self.actual_improve = actual_improve
        self.exchange_msg = exchange_msg

    def __repr__(self):
        return str(self.__dict__)


class Generation_Data():
    
    def __init__(self, tid: str, gen_num: int):
        self.tid = tid
        self.gen_num = gen_num
        self.transactions: List[Order_Data] = []

    def __repr__(self):
        return str(self.__dict__)


class STGP_Trader(Trader):
    
    def __init__(self, tid, balance, time, trading_func):
        super().__init__("STGP", tid, balance, time)

        # trading function from STGP tree (calculates improvement on customer order).
        self.trading_func = trading_func

        # profit tracking
        self.last_evolution = 0.0
        self.current_gen = 0
        self.profit_since_evolution = 0.0
        self.generational_profits = []

        # Exponential Moving Average
        self.ema = None
        self.nLastTrades = 5
        self.ema_param = 2 / float(self.nLastTrades + 1)

        # Stat tracking
        self.all_gens_data = []
        self.current_gen_data = Generation_Data(tid, self.current_gen)

    
    def del_cust_order(self, cust_order_id, verbose):
        super().del_cust_order(cust_order_id, verbose)

    def get_profit(self, time):
        """ Gets the profit of the current generation. Used for gp evaluation. """
        if time == 0: return 0
        return self.profit_since_evolution

    def reset_gen_profits(self):
        """ Called after the expr has been updated due to evolution. This is necessary to evaluate a generation. """
        self.generational_profits.append(self.profit_since_evolution)
        self.profit_since_evolution = 0
        self.current_gen += 1

        # logging
        self.all_gens_data.append(self.current_gen_data)
        self.current_gen_data = Generation_Data(self.tid, self.current_gen)

    def _update_ema(self, price):
        """ Update exponential moving average indicator for the trader. """
        if self.ema == None: self.ema = price
        else: self.ema = self.ema_param * price + (1 - self.ema_param) * self.ema

    def getorder(self, time, countdown, lob, verbose):
        """ Called by the market session to get an order from this trader. """
        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            self.limit = self.orders[0].price
            self.job = self.orders[0].atype

            improvement = 0
            
            # calculate improvement on customer order via STGP function
            if self.ema != None and lob['bids']['bestp'] != None and lob['asks']['bestp'] != None:
                improvement = self.trading_func(self.ema, lob['bids']['bestp'], lob['asks']['bestp'], time, countdown)
            # resets negative improvements
            if improvement < 0:
                improvement = 0
            if verbose:
                print(f"trader: {self.tid}, limit price: {self.limit}, improvement found: {improvement}")

            # print(f"trader: {self.tid}, limit price: {self.limit}, improvement found: {improvement}")

            # buys for less, sells for more
            if self.job == 'Bid':
                quoteprice = int(self.limit - improvement)
            elif self.job == 'Ask':
                quoteprice = int(self.limit + improvement)

            order = Order(self.tid, self.job, "LIM", quoteprice, 
                          self.orders[0].qty, time, None, -1)

            self.price = quoteprice
            self.lastquote = order

        return order

    def respond(self, time, lob, trade, verbose):
        """ Called by the market session to notify trader of LOB updates. """
        if (trade != None):
            self._update_ema(trade["price"]) # update EMA

    def bookkeep(self, msg, time, verbose):
        if msg.event == "FILL":
            improvement = abs(msg.trns[0]["Price"] - self.orders[0].price)
            self.profit_since_evolution += improvement

            # for logging: customer price, trans price, attemted improvement, actual improvement, msg
            posted_improvement = abs(self.lastquote.price - self.limit)
            new_order_data = Order_Data(self.limit, msg.trns[0]["Price"], 
                                        posted_improvement, improvement, msg)
            self.current_gen_data.transactions.append(new_order_data)
        
        super().bookkeep(msg, time, verbose)

    def get_gen_profits(self):
        """ to be called at the end of experiment """
        ps = self.generational_profits
        ps.append(self.profit_since_evolution)
        return ps
