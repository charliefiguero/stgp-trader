from deap import gp, creator, base, tools


class Pset_and_Toolbox():

    def __init__(self):

        ########## PRIMITIVE SET ############

        # initialise pset
        self.pset = gp.PrimitiveSetTyped("main", [float, float, float, float, float, float], float)
        # inputs to the tree
        pset.renameArguments(ARG0="ema_ind")
        pset.renameArguments(ARG1="best_same") # same orderbook side
        pset.renameArguments(ARG2="best_opp") # opposite orderbook side
        pset.renameArguments(ARG3="time")
        pset.renameArguments(ARG4="countdown")
        pset.renameArguments(ARG5="cust_price")

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
        pset.addEphemeralConstant(ephemeral_name, lambda: random.randint(-10, 10), float)



        # ####### CREATOR CLASSES #######

        # create fitness class
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # create individual class
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, trader_id = None)



        ######## TOOLBOX #########

        # create toolbox
        toolbox = base.Toolbox()

        # init tools
        toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
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
