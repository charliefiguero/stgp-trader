""" GP_Entity will initialise multiple of these and give them each an improvement function dictated by the tree of the individual."""

import sys
import random

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader


class GP_Trader(Trader):
    
    def __init__(self, tree):
        val = 1

