import sys
import os
import inspect
import random
import operator
import numpy
import datetime
from datetime import datetime
from typing import NewType
from time import sleep
from statistics import mean, stdev
import jsonpickle
import pprint
from operator import attrgetter
from copy import copy

from deap import gp, creator, base, tools, algorithms
import pygraphviz as pgv

from BSE2_msg_classes import Assignment, Order, ExchMsg
from BSE2_trader_agents import Trader
from BSE2_Entity import Entity
from STGP_Trader import STGP_Trader
import experiment_setup


def if_then_else(inputed, output1, output2):
    return output1 if inputed else output2


class STGP_Entity(Entity):

    def __init__(self, id, init_balance, job, duration):
        super().__init__(id, init_balance, 0, {})

        if job != 'BUY' and job != 'SELL':
            raise ValueError('Tried to initialise entity with unknown job type. '
            'Should be either \'BUY\' or \'SELL\'.')
        self.job = job

        self.pset, self.toolbox = self.create_deap_toolbox_and_pset()
        self.exprs = [] # gp trees that are evolved

        # for logging
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", lambda xs : mean([x[0] for x in xs]))
        self.stats.register("std", lambda xs: stdev([x[0] for x in xs]))
        self.stats.register("min", min)
        self.stats.register("max", max)
        self.gen_records = []
        self.hall_of_fame = tools.HallOfFame(1)
        self.prv_exprs = []
        self.best_exprs = [] # I don't know why hall of fame doesn't save as strings? refactor this

        self.traders = {} # traders are passed compiled exprs
        self.traders_count = 0

        self.duration = duration
        self.NUM_GENS = experiment_setup.NUM_GENS
        self.EVAL_TIME = duration / self.NUM_GENS # evolves the pop every x seconds
        self.last_update = 0

    def create_deap_toolbox_and_pset(self):
        # initialise pset
        pset = gp.PrimitiveSetTyped("main", [float, float, float, float, float], float)
        # inputs to the tree
        pset.renameArguments(ARG0="ema_ind")
        pset.renameArguments(ARG1="best_ask")
        pset.renameArguments(ARG2="best_bid")
        pset.renameArguments(ARG3="time")
        pset.renameArguments(ARG4="countdown")
        # float operations
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
        # bloat control
        # max height == 17 (as specificed by Koza, 1989)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        return pset, toolbox

    def init_traders(self, n: int, balance: int, time: float):
        self.exprs = self.toolbox.population(n)

        self.prv_exprs.append(self.exprs.copy())

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
        for expr in self.exprs:
            expr.fitness.values = (self.evaluate_expr(expr, time), )
        return map(lambda x: x.fitness.values[0], self.exprs)

    def evolve_population(self, time):
        print(f"Evolving population for {self.lei}, generation: {int(time / self.EVAL_TIME)}")

        profits = [tr.get_profit(time) for tr in self.traders.values()]
        print(f'generational profits: {sum(profits)}\n')
        
        # Evaluate the entire population
        fitnesses = self.evaluate_population(time)
        
        # update the hall of fame
        self.hall_of_fame.update(self.exprs)
        best_ind = sorted(self.exprs, key=attrgetter('fitness.values'), reverse=True)[0]
        print('best_ind from stgp enittiy', best_ind, 'type:', type(best_ind))
        # print('best ind type:', type(best_ind))
        # best_fitness = best_ind.fitness.values[0]
        self.best_exprs.append(copy(best_ind)) # TODO: refactor to just use the hall of fame?

        # statistics
        record = self.stats.compile(self.exprs)
        self.gen_records.append(record)
        print(record)

        # Select the next generation individuals
        offspring = self.toolbox.select(self.exprs, len(self.exprs))
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        # [::2] means nothing for first and second argument and jump by 2.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < experiment_setup.CXPB:
                # print(f"about to mate: Child1: {child1}   Child2: {child2}")
                # draw_expr(child1, "child1 pre")
                # draw_expr(child2, "child2 pre")
                # print(child1)
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                # print(f"mated: Child1: {child1}   Child2: {child2}")
                # draw_expr(child1, "child1 post")
                # draw_expr(child2, "child2 post")
                
        for mutant in offspring:
            if random.random() < experiment_setup.MUTPB:
                # print(f"about to mutate: {mutant} \n")
                # draw_expr(mutant, f"tree pre {mutant}")
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
                # print(f"mutant: {mutant}")
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

        # sleep(1)

        # reset trader stats for profit eval 
        for trader in self.traders.values():
            trader.reset_gen_profits()

        # log new exprs
        self.prv_exprs.append(self.exprs.copy())

        print('finished evolving\n')
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
        if time - self.last_update > self.EVAL_TIME:
            self.evolve_population(time)
            self.last_update = time

    def total_gen_profits(self):
        num_gen = len(list(self.traders.values())[0].generational_profits) + 1

        all_profits = []
        for i in range(num_gen-1):
            gen_profits = []
            for t in self.traders.values():
                gen_profits.append(t.generational_profits[i])
            all_profits.append(gen_profits)

        final_gen_profits = []
        for t in self.traders.values():
            final_gen_profits.append(t.profit_since_evolution)

        all_profits.append(final_gen_profits)

        output = [(count, sum(x)) for count, x in enumerate(all_profits)]
        print(f"total gen profits...\n{output}\n")
        return output

    def write_total_gen_profits(self):
        """ called at the end of experiment """

        data = {}
        data["num_gens"] = self.NUM_GENS
        data["num_traders"] = len(self.traders)
        traders_data = {}
        for tr in self.traders.values():
            tr.all_gens_data.append(tr.current_gen_data)
            traders_data[tr.tid] = tr.all_gens_data 
        data["traders_data"] = traders_data

        now = datetime.now() 
        with open('stgp_csvs/improvements/' + str(now), 'w') as outfile:
            outfile.write(jsonpickle.encode(data, indent=4))

    def write_gen_records(self):
        now = datetime.now()
        with open('stgp_csvs/gen_records/' + str(now), 'w') as outfile:
            outfile.write(jsonpickle.encode(self.gen_records, indent=4))

    def write_hof(self):
        now = datetime.now()
        with open('stgp_csvs/hall_of_fame/' + str(now), 'w') as outfile:
            for tree in self.hall_of_fame:
                output = gp.PrimitiveTree(tree)
                outfile.write(jsonpickle.encode(output, indent=4))


if __name__ == "__main__":
    
    e = STGP_Entity("STGP0", 100, 'BUY', 1000)
    e.init_traders(10, 100, 0.0)

    list(e.traders.values())[0].get_gen_profits()
    print('hello')

