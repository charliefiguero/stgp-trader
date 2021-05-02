""" GP_Entity will initialise multiple of these and give them each an improvement function dictated by the tree of the individual."""

import sys
import random

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader


class STGP_Trader(Trader):
    
    def __init__(self, tid, balance, time, trading_func):
        super().__init__("STGP", tid, balance, time)

        # trading function from STGP tree (calculates improvement on customer order).
        self.trading_func = trading_func
        self.last_evolution = 0

        # Exponential Moving Average
        self.ema = None
        self.nLastTrades = 5
        self.ema_param = 2 / float(self.nLastTrades + 1)

    def getorder(self, time, countdown, lob, verbose):
        """ Called by the market session to get an order from this trader. """
        if verbose:
            print('STGP Trader getorder:')

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            self.limit = self.orders[0].price
            self.job = self.orders[0].atype

            improvement = 0
            
            # calculate improvement on customer order via STGP function
            if self.ema != None:
                improvement = self.trading_func(self.ema)
                
            # resets negative improvements
            if improvement < 0:
                improvement = 0

            # print(f"trader: {self.tid}, limit price: {self.limit}, improvement found: {improvement}")

            if verbose:
                print(f"improvement: {improvement}")

            if self.job == 'Bid':
                quoteprice = int(self.limit - improvement)
            elif self.job == 'Ask':
                quoteprice = int(self.limit + improvement)

            self.price = quoteprice

            order = Order(self.tid, self.job, "LIM", quoteprice, 
                          self.orders[0].qty, time, None, -1)

            self.lastquote = order

            if verbose:
                print(f"stgp trader making order with price: {quoteprice}")
        return order

    def _update_ema(self, price):
        """ Update exponential moving average indicator for the trader. """
        if self.ema == None: self.ema = price
        else: self.ema = self.ema_param * price + (1 - self.ema_param) * self.ema

    def respond(self, time, lob, trade, verbose):
        """ Called by the market session to notify trader of LOB updates. """
        if (trade != None):
            self._update_ema(trade["price"]) # update EMA
