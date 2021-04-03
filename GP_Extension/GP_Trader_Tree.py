import sys
import random
import math
import operator
import numpy

from deap import gp, creator, base, tools, algorithms
from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader

class GP_Trader_Tree(Trader):
    