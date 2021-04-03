import sys
import os
import inspect
import random
import math
import operator
import numpy

from deap import gp, creator, base, tools, algorithms

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader
from BSE2_Entity import Entity

class STGP_Entity(Entity):
    Trader()
    pass