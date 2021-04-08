import sys
import os
import inspect
import random
import operator
import numpy
import datetime

from deap import gp, creator, base, tools, algorithms
import pygraphviz as pgv

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader
from BSE2_Entity import Entity
from STGP_Trader import STGP_Trader


def if_then_else(inputed, output1, output2):
    return output1 if inputed else output2

def draw_expr(expr):
    nodes, edges, labels = gp.graph(expr)

    ### Graphviz Section ###
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    # saves to tree.pdf
    now = datetime.datetime.now()
    # print(f"Current time: {now}\n")
    g.draw(f"trees/tree {now}.pdf")


class STGP_Entity(Entity):

    def __init__(self, id, init_balance):
        super().__init__(id, init_balance, 0, {})
        self.pset, self.toolbox = self.create_deap_toolbox_and_pset()

        self.exprs = []

    def create_deap_toolbox_and_pset(self):
        # initialise pset
        pset = gp.PrimitiveSetTyped("main", [float], float)
        pset.renameArguments(ARG0="ema_ind")
        # integer operations
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        # conditional operations 
        pset.addPrimitive(if_then_else, [bool, float, float], float)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.eq, [float, float], bool)
        # boolean terminals
        pset.addTerminal(True, bool)
        pset.addTerminal(False, bool)
        # ephemeral terminals
        pset.addEphemeralConstant("qwer", lambda: random.randint(-10, 10), float)

        # create individual class
        creator.create("Individual", gp.PrimitiveTree)

        # create toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        return pset, toolbox

    def init_traders(self, n, balance, time):
        self.exprs = self.toolbox.population(n)

        for count, expr in enumerate(self.exprs):
            trading_function = gp.compile(gp.PrimitiveTree(expr), self.pset)
            self.traders.append(STGP_Trader(count, balance, time, trading_function))
        

if __name__ == "__main__":

    
    e = STGP_Entity(0, 100)
    e.init_traders(10, 100, datetime.datetime.now())

    # exprs = e.toolbox.population(10)
    # draw_expr(exprs[0])
    # a_tree = gp.PrimitiveTree(exprs[0])
    # print(a_tree)
    # function = gp.compile(a_tree, e.pset)
    # print(function(1))


    # TODO create traders
    # TODO initialise traders in STGP_Entity
    # TODO register traders on the exchange
