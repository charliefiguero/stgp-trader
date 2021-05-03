import sys
import os
import inspect
import random
import operator
import numpy
import datetime
from typing import NewType
from time import sleep

from deap import gp, creator, base, tools, algorithms
import pygraphviz as pgv

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader
from BSE2_Entity import Entity
from STGP_Trader import STGP_Trader


def if_then_else(inputed, output1, output2):
    return output1 if inputed else output2

def draw_expr(expr, name=None):
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
    if not name == None:
        g.draw(f"trees/tree {name}.pdf")
    else:
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
        self.traders_count = 0

        self.eval_time = 50
        self.last_update = 0

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
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, trader_id = None)

        # create toolbox
        toolbox = base.Toolbox()
        # init tools
        toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # evolution tools
        toolbox.register("evaluate", self.evaluate_expr)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        return pset, toolbox

    def init_traders(self, n: int, balance: int, time: float):
        self.exprs = self.toolbox.population(n)

        for count, expr in enumerate(self.exprs):
            # traders never get the same id
            count += self.traders_count

            trading_function = gp.compile(gp.PrimitiveTree(expr), self.pset)
            tname = 'STGP_%02d' % count

            # update tname for customer_orders shenanigans
            if self.job == 'BUY':
                tname = 'B' + tname
            elif self.job == "SELL":
                tname = 'S' + tname

            expr.trader_id = tname
            self.traders[tname] = STGP_Trader(tname, balance, time, trading_function)

        self.traders_count += len(self.exprs)

    # used for self.toolbox.evaluate
    def evaluate_expr(self, improvement_expr, time):
        return self.traders[improvement_expr.trader_id].get_profit(time)

    def evaluate_population(self, time):
        """ calculate fitnesses for the population """
        return map(lambda x: self.toolbox.evaluate(x, time), self.exprs)

    def evolve_population(self, time):
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        print("Evolving population")
        
        # Evaluate the entire population
        fitnesses = self.evaluate_population(time)
        # Select the next generation individuals
        offspring = self.toolbox.select(self.exprs, len(self.exprs))
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        # [::2] means nothing for first and second argument and jump by 2.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                print(f"about to mate: Child1: {child1}   Child2: {child2}")
                # draw_expr(child1, "child1 pre")
                # draw_expr(child2, "child2 pre")
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                print(f"mated: Child1: {child1}   Child2: {child2}")
                # draw_expr(child1, "child1 post")
                # draw_expr(child2, "child2 post")
                
        print()

        for mutant in offspring:
            if random.random() < MUTPB:
                print(f"about to mutate: {mutant} \n")
                # draw_expr(mutant, f"tree pre {mutant}")
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
                print(f"mutant: {mutant}")
                # draw_expr(mutant, f"tree post {mutant}")

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(lambda x: self.toolbox.evaluate(x, time), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            # print(f"fit: {fit}")
            # print(f"values: {ind.fitness.values}")
            ind.fitness.values = (fit,)

        # The population is entirely replaced by the offspring
        self.exprs[:] = offspring
        self.update_trader_expr(time)

        sleep(1)

        # reset trader stats for logging
        for trader in self.traders.values():
            trader.reset_gen_profits()

        return self.exprs

    def update_trader_expr(self, time):
        if not len(self.exprs) == len(self.traders):
            raise ValueError(f"len(exprs): {len(self.exprs)} != len(traders): {len(self.traders)}")

        for count, trader in enumerate(self.traders.values()):
            expr = self.exprs[count]
            trader.trading_func = gp.compile(gp.PrimitiveTree(expr), self.pset)
            expr.trader_id = trader.tid
            trader.last_evolution = time

    def entity_update(self, time):
        """ Called by market session every time tick. """
        if time - self.last_update > self.eval_time:
            self.evolve_population(time)
            self.last_update = time


if __name__ == "__main__":
    
    e = STGP_Entity(0, 100, 'BUY')
    e.init_traders(10, 100, 0.0)

    print("traders pre evolution")
    for expr in e.exprs:
        print(expr)

    print()

    for expr in e.exprs:
        print(expr.trader_id)

    e.evolve_population(0)
    print()

    for expr in e.exprs:
        print(expr.trader_id)

    print("expr post evolution")
    for expr in e.exprs:
        print(expr)

    # for expr in e.exprs:
    #     print(expr)
    # e.exprs[0].fitness.values = e.evaluate_trader(e.exprs[0])

    # print(e.exprs[0].trader_id)
    # print(e.exprs[1].trader_id)
    # print(e.exprs[2].trader_id)
    # for expr in e.exprs:
    #     print(expr.trader_id)

    # print(len(e.exprs))
    # print(len(e.traders))
    # print(e.traders.keys())





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
