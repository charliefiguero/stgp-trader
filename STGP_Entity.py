import sys
import os
import inspect
import random
import operator
import numpy
import datetime
from typing import NewType

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

    def __init__(self, id, init_balance, job):
        super().__init__(id, init_balance, 0, {})

        if job != 'BUY' and job != 'SELL':
            raise ValueError('Tried to initialise entity with unknown job type. '
            'Should be either \'BUY\' or \'SELL\'.')
        self.job = job

        self.pset, self.toolbox = self.create_deap_toolbox_and_pset()
        self.exprs = []
        self.traders = {}

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

        # create fitness class
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # create individual class
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        # create toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_trader)

        return pset, toolbox

    def init_traders(self, n: int, balance: int, time: float):
        self.exprs = self.toolbox.population(n)

        for count, expr in enumerate(self.exprs):
            trading_function = gp.compile(gp.PrimitiveTree(expr), self.pset)
            tname = 'STGP_%02d' % count

            # update tname for customer_orders shenanigans
            if self.job == 'BUY':
                tname = 'B' + tname
            elif self.job == "SELL":
                tname = 'S' + tname

            self.traders[tname] = STGP_Trader(tname, balance, time, trading_function)

    def evaluate_trader(self, trader):
        return trader.profitpertime

    def evaluate_population(self):
        """ calculate fitnesses for the population """
        return map(toolbox.evaluate, self.traders.values)

    def evolve_population(self):
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        # Evaluate the entire population
        fitnesses = evaluate_population()
        # Select the next generation individuals
        offspring = toolbox.select(self.pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        return pop
        

if __name__ == "__main__":
    
    e = STGP_Entity(0, 100, 'BUY')
    e.init_traders(10, 100, 0.0)

    # for trader in e.traders:
    #     print(trader)

    print()

    # for expr in e.exprs:
    #     print(expr)
    # e.exprs[0].fitness.values = e.evaluate_trader(e.exprs[0])
    print(e.exprs[0].count)
    print(dir(e.exprs[0]))




    # trader = e.traders[0]
    # trader._update_ema(10)
    # draw_expr(e.exprs[0])

    # order = Assignment("CUS", "123", 'Bid', 'LIM', 
    #                    10, 1, 1, None, 1)
    # trader.add_cust_order(order, verbose=True)
    # print(e.traders[0].getorder(0, 0, [0], True))



    # exprs = e.toolbox.population(10)
    # draw_expr(exprs[0])
    # a_tree = gp.PrimitiveTree(exprs[0])
    # print(a_tree)
    # function = gp.compile(a_tree, e.pset)
    # print(function(1))

    # TODO register traders on the exchange
    # TODO trader evaluation and evolution
