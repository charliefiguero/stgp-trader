# general
END_TIME = 100.0
# must be atleast 1 - needs refactoring
BUYERS_SPEC = [('GVWY', 1)] 
SELLERS_SPEC = [('GVWY', 101)]

# STGP variables
NUM_TRADERS_PER_ENTITY = 100
STGP_TRADER_STARTING_BALANCE = 100
NUM_GENS = 20
evolving = True

# probability 1 <= p <= 0
CXPB = 0.3 # crossover probability
MUTPB = 0.1 # mutation probability

TOURNSIZE = 10

# rest of experiment setup in BSE2 : main()
# eg. duration, other traders, ...
