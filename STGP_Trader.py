""" GP_Entity will initialise multiple of these and give them each an improvement function dictated by the tree of the individual."""

import sys
import random

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader


class STGP_Trader(Trader):
    
    def __init__(self, tid, balance, time, trading_func):
        super().__init__("stgp_trader", tid, balance, time)
        self.trading_func = trading_func

    def getorder(self, time, countdown, lob, verbose):
        pass

    def respond(self, time, lob, trade, verbose):
        pass

    def ema_ind(self):
        """ exponential moving average indicator for the trader """
        ema = 1
        return ema
    


