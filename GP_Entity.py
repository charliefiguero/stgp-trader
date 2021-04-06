import sys
import os
import inspect
import random
import operator
import numpy

from deap import gp, creator, base, tools, algorithms
import pygraphviz as pgv

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader
from BSE2_Entity import Entity


def if_then_else(inputed, output1, output2):
    return output1 if inputed else output2


class STGP_Entity(Entity):

    def __init__(self, id, init_balance):
        super().__init__(id, init_balance, 0, {})

        self.pset = self.create_primitive_set()
        self.pop_size = 1
        # self.traders.update(self.fullgen_initial_pop())

        self.time_since_last_trader_eval = 0

    def create_primitive_set(self):
        """ Creates the primitive set for the STGP traders: the building blocks of the trees. Called once at init of STGP_Entity. """

        pset = gp.PrimitiveSetTyped("main", [int, bool], int)

        # integer operations
        pset.addPrimitive(operator.add, [int, int], int)
        pset.addPrimitive(operator.sub, [int, int], int)

        # boolean operations
        # pset.addPrimitive(operator.and_, [bool, bool], bool)
        # pset.addPrimitive(operator.or_, [bool, bool], bool)
        # pset.addPrimitive(operator.xor, [bool, bool], bool)
        # pset.addPrimitive(operator.not_, bool, bool)

        # conditional operations 
        pset.addPrimitive(if_then_else, [bool, float, float], float)
        pset.addPrimitive(operator.lt, [int, int], bool)
        pset.addPrimitive(operator.gt, [int, int], bool)
        pset.addPrimitive(operator.eq, [int, int], bool)

        # basic terminals
        # pset.addTerminal(1, bool)
        # pset.addTerminal(0, bool)

        # ephemeral terminals
        pset.addEphemeralConstant("integer", lambda: random.randint(-10, 10), int)

        # variables - enter these into the tree then replace them with references oncde the tree has been instantiated.
        # eg. pset.addTerminal("orderbookdata", var)
        # terminal orderbookdata = limitorderbook[0]

        return pset

    def convert_tree_variable_to_reference(self):
        pass

    def fullgen_initial_pop(self, size):
        pop = gp.genFull(self.pset, 2, 3)
        return pop

    def evaluate_trader(self, trader):
        """ evaluates the traders based on their profit over a given timeframe """
        pass

    def create_var_pop(self):
        """ vary the population according to genetic programming. """
        # deap/algorithms.py :: varOr()
        pass

    def evolve_traders(self):
        pass


if __name__ == "__main__":
    e = STGP_Entity(0, 100)
    expr = e.fullgen_initial_pop(1)
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
    g.draw("trees/tree.pdf")



    # TODO print output of tree