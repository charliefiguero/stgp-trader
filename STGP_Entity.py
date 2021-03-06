import random
import operator
from datetime import datetime
from statistics import mean, stdev
import jsonpickle
from operator import attrgetter
import pickle
import json
from copy import deepcopy
import math

from deap import gp, creator, base, tools, algorithms

from BSE2_Entity import Entity
from STGP_Trader import STGP_Trader
import experiment_setup
import BSE2_sys_consts


def if_then_else(inputed, output1, output2):
    return output1 if inputed else output2

def div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


class STGP_Entity(Entity):

    def __init__(self, id, init_balance, job, duration, offspring_file):
        super().__init__(id, init_balance, 0, {})

        if job != 'BUY' and job != 'SELL':
            raise ValueError('Tried to initialise entity with unknown job type. '
            'Should be either \'BUY\' or \'SELL\'.')
        self.job = job

        self.pset, self.toolbox = self.create_deap_toolbox_and_pset()
        self.exprs = [] # gp trees that are evolved
        self.offspring_file = offspring_file
        # reset offspring_file
        with open(self.offspring_file, 'w') as f:
            f.write('')
        self.gen = 1

        # for logging
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", lambda xs : mean([x[0] for x in xs]))
        self.stats.register("std", lambda xs: stdev([x[0] for x in xs]))
        self.stats.register("min", min)
        self.stats.register("max", max)
        self.stats.register
        self.gen_records = []
        self.hall_of_fame = tools.HallOfFame(1)
        # self.prv_exprs = []

        self.best_ind_fname = str(datetime.now())
        with open('stgp_csvs/generational_best/' + self.best_ind_fname, 'w') as outfile:
            outfile.write('')

        self.traders = {} # traders are passed compiled exprs
        self.traders_count = 0

        self.duration = duration
        self.NUM_GENS = experiment_setup.NUM_GENS
        self.EVAL_TIME = duration / self.NUM_GENS # evolves the pop every x seconds
        self.last_update = 0

    def create_deap_toolbox_and_pset(self):
        # initialise pset
        pset = gp.PrimitiveSetTyped("main", [float, float, bool, bool, float, float, float, float, float, float, float], float)
        # inputs to the tree
        pset.renameArguments(ARG0="stub")
        pset.renameArguments(ARG1="ltp") # same orderbook side
        pset.renameArguments(ARG2="same_present") # opposite orderbook side
        pset.renameArguments(ARG3="opp_present")
        pset.renameArguments(ARG4="best_same") # time till end of trading period
        pset.renameArguments(ARG5="best_opp") # customer limit price
        pset.renameArguments(ARG6="worst_same") # system min for buy, max for sell
        pset.renameArguments(ARG7="worst_opp") # system max for buy, min for sell
        pset.renameArguments(ARG8="rand")
        pset.renameArguments(ARG9="countdown")
        pset.renameArguments(ARG10="time")
        # float operations
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(div, [float, float], float)
        # conditional operations 
        pset.addPrimitive(if_then_else, [bool, float, float], float)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.eq, [float, float], bool)
        # boolean terminals
        pset.addTerminal(True, bool)
        pset.addTerminal(False, bool)

        # ephemeral terminals
        # "Ephemerals with different functions should be named differently, even between psets."
        # hacked :()
        if self.job == 'BUY':
            pset.addEphemeralConstant('ephemeral_b', lambda: random.randint(-10, 10), float)
        elif self.job == 'SELL':
            pset.addEphemeralConstant('ephemeral_s', lambda: random.randint(-10, 10), float)
        else:
            raise ValueError('Entity has unknown job type. '
            'Should be either \'BUY\' or \'SELL\'.')


        # ######## CREATOR CLASSES ########

        # another hack for DEAP :(
        if self.job == 'BUY':
            # create fitness class
            creator.create("FitnessMax_BUY", base.Fitness, weights=(1.0,))
            # create individual class
            creator.create("Individual_BUY", gp.PrimitiveTree, fitness=creator.FitnessMax_BUY, trader_id = None)

        elif self.job == 'SELL':
            # create fitness class
            creator.create("FitnessMax_SELL", base.Fitness, weights=(1.0,))
            # create individual class
            creator.create("Individual_SELL", gp.PrimitiveTree, fitness=creator.FitnessMax_SELL, trader_id = None)


        # create toolbox
        toolbox = base.Toolbox()
        # init tools
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

        # another hack for DEAP :(
        if self.job == 'BUY':
            toolbox.register("individual", tools.initIterate, creator.Individual_BUY, toolbox.expr)
        elif self.job == 'SELL':
            toolbox.register("individual", tools.initIterate, creator.Individual_SELL, toolbox.expr)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # evolution tools
        toolbox.register("evaluate", self.evaluate_expr)
        toolbox.register("select", tools.selTournament, tournsize=experiment_setup.TOURNSIZE)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        # bloat control
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        return pset, toolbox

    def init_traders(self, n: int, balance: int, time: float):

        with open('trader_conversions.json', 'r') as infile:
            traders = json.load(infile)
            shvr = traders['SHVR']
            zic = traders['ZIC']
            gvwy = traders['GVWY']

            zic_improved = traders["ZIC_improved"]
            gvwy_improved = traders["GVWY_improved"]
            shvr_improved = traders["SHVR_improved"]

        # load in prebuilt traders 
        if self.job == 'BUY':
            bzic_ratio = 0
            bshvr_ratio = 0
            bgvwy_ratio = 1
            brand_ratio = 0
            improved_ratio = 1
            num_bzic = math.floor(int(n * bzic_ratio))
            num_bshvr = math.floor(int(n * bshvr_ratio))
            num_bgvwy = math.floor(int(n * bgvwy_ratio))
            num_brand = n - num_bzic - num_bshvr - num_bgvwy
            num_improved = n

            bzic = [creator.Individual_BUY(gp.PrimitiveTree.from_string(zic, self.pset)) for x in range(num_bzic)]
            bshvr = [creator.Individual_BUY(gp.PrimitiveTree.from_string(shvr, self.pset)) for x in range(num_bshvr)]
            bgvwy = [creator.Individual_BUY(gp.PrimitiveTree.from_string(gvwy, self.pset)) for x in range(num_bgvwy)]
            brand = [self.toolbox.individual() for x in range(num_brand)]
            bimproved = [creator.Individual_BUY(gp.PrimitiveTree.from_string(gvwy_improved, self.pset)) for x in range(num_improved)]

            loaded_inds = bimproved
            # loaded_inds = bzic + bshvr + bgvwy 
            # loaded_inds = bzic + bshvr + bgvwy + brand
            # loaded_inds = bzic 
            # loaded_inds = bshvr 
            # loaded_inds = bgvwy
            # loaded_inds = bgvwy
        elif self.job == 'SELL':
            szic_ratio = 0
            sshvr_ratio = 0
            sgvwy_ratio = 1
            srandom_ratio = 0
            improved_ratio = 1
            num_szic = math.floor(int(n * szic_ratio))
            num_sshvr = math.floor(int(n * sshvr_ratio))
            num_sgvwy = math.floor(int(n * sgvwy_ratio))
            num_srand = n - num_szic - num_sshvr - num_sgvwy
            num_improved = n

            szic = [creator.Individual_SELL(gp.PrimitiveTree.from_string(zic, self.pset)) for x in range(num_szic)]
            sshvr = [creator.Individual_SELL(gp.PrimitiveTree.from_string(shvr, self.pset)) for x in range(num_sshvr)]
            sgvwy = [creator.Individual_SELL(gp.PrimitiveTree.from_string(gvwy, self.pset)) for x in range(num_sgvwy)]
            simproved = [creator.Individual_SELL(gp.PrimitiveTree.from_string(gvwy_improved, self.pset)) for x in range(num_improved)]
            srand = [self.toolbox.individual() for x in range(num_srand)]


            loaded_inds = simproved
            # loaded_inds = szic + sshvr + sgvwy 
            # loaded_inds = szic + sshvr + sgvwy + srand 
            # loaded_inds = szic
            # loaded_inds = sshvr
            # loaded_inds = sgvwy

        # self.exprs = self.toolbox.population(n)
        self.exprs = loaded_inds

        # self.prv_exprs.append(deepcopy(self.exprs))

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
        # orderprices = []
        # for tr in self.traders.values():
        #     orderprices.extend(tr.orderprices)
        # averageop = mean(orderprices)
        # print(self.lei, averageop)
        # print(self.lei, orderprices)

        if not experiment_setup.evolving:
            print(f'generation: {int(time / self.EVAL_TIME)}')
            profits = [tr.get_profit(time) for tr in self.traders.values()]
            print(f'generational profits: {sum(profits)}')

            offspring = deepcopy(self.exprs)
            self.exprs[:] = offspring
            self.update_trader_expr(time)

            for trader in self.traders.values():
                trader.reset_gen_profits()
            return

        self.gen += 1
        self.write_to_offspring_file(self.gen, time)

        # Normal evolution

        print(f"Evolving population for {self.lei}, generation: {int(time / self.EVAL_TIME)}")

        profits = [tr.get_profit(time) for tr in self.traders.values()]
        print(f'generational profits: {sum(profits)}')
        
        # Evaluate the entire population
        fitnesses = self.evaluate_population(time)
        
        # update the hall of fame
        self.hall_of_fame.update(self.exprs)

        best_ind = sorted(self.exprs, key=attrgetter('fitness.values'), reverse=True)[0]
        with open('stgp_csvs/generational_best/' + self.best_ind_fname, 'a') as outfile:
            outfile.write(str(best_ind)+'\n')
        print('best_ind from generation:', best_ind)

        # statistics
        record = self.stats.compile(self.exprs)
        self.gen_records.append(record)
        print(record)

        # Select the next generation individuals
        offspring = self.toolbox.select(self.exprs, len(self.exprs))
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))
        
        # print('old')
        # print(list(map(lambda x : x.fitness.values, self.exprs)))
        # print('new')
        # print(list(map(lambda x : x.fitness.values, offspring)))

        # Apply crossover and mutation on the offspring
        # [::2] means nothing for first and second argument and jump by 2.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < experiment_setup.CXPB:
                # print(f"about to mate: Child1: {child1}   Child2: {child2}")
                # print(child1)
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                # print(f"mated: Child1: {child1}   Child2: {child2}")s
                
        for mutant in offspring:
            if random.random() < experiment_setup.MUTPB:
                # print(f"about to mutate: {mutant} \n")
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
                # print(f"mutant: {mutant}")

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(lambda x: self.toolbox.evaluate(x, time), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            # print(f"fit: {fit}")
            # print(f"values: {ind.fitness.values}")
            ind.fitness.values = (fit,)

        # The population is entirely replaced by the offspring
        self.exprs[:] = offspring

        # print(gp.PrimitiveTree(offspring[0]))

        self.update_trader_expr(time)

        # reset trader stats for profit eval 
        for trader in self.traders.values():
            trader.reset_gen_profits()

        # log new exprs
        # self.prv_exprs.append(self.exprs.copy())

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
        with open('stgp_csvs/improvements/' + str(now) + self.lei + '.json', 'w') as outfile:
            outfile.write(jsonpickle.encode(data, indent=4))

    def write_gen_records(self):
        now = datetime.now()
        with open('stgp_csvs/gen_records/' + str(now) + self.lei + '.json', 'w') as outfile:
            outfile.write(jsonpickle.encode(self.gen_records, indent=4))

    def write_hof(self):
        now = datetime.now()
        with open('stgp_csvs/hall_of_fame/' + str(now) + self.lei, 'wb') as outfile:
            for tree in self.hall_of_fame:
                output = gp.PrimitiveTree(tree)
                pickle.dump(output, outfile)

    def write_to_offspring_file(self, gen, time):
        withprofit = [(self.evaluate_expr(expr, time), str(gp.PrimitiveTree(expr)), expr.trader_id) for expr in self.exprs]
        withprofit.sort(reverse=True)
        with open(self.offspring_file, 'a') as f:
            f.write(f"Gen: {gen}\n")
            for e in withprofit:
                f.write(str(e[0]) + ": " + e[2] + " : " + str(e[1]) + '\n')
            f.write("\n")


if __name__ == "__main__":
    
    e = STGP_Entity("STGP0", 100, 'BUY', 1000)

    with open('trader_conversions.json', 'r') as infile:
        traders = json.load(infile)
        shvr = traders['SHVR']
        test = gp.PrimitiveTree.from_string(shvr, e.pset)
        ind = creator.Individual(test)

        print(ind)
        # test = gp.PrimitiveTree.from_string


    # e.init_traders(10, 100, 0.0)

    # list(e.traders.values())[0].get_gen_profits()
    # print('hello')

