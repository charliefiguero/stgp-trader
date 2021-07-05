# general
END_TIME = 100000.0
# must be atleast 1 - needs refactoring
BUYERS_SPEC = [('ZIP', 100),('ZIC', 100),('GVWY', 100),('SHVR', 50)] 
SELLERS_SPEC = [('ZIP', 100),('ZIC', 100),('GVWY', 100),('SHVR', 150)]

# STGP variables
NUM_TRADERS_PER_ENTITY = 50
STGP_TRADER_STARTING_BALANCE = 100
NUM_GENS = 40
evolving = True

# probability 1 <= p <= 0
CXPB = 0.3 # crossover probability
MUTPB = 0.1 # mutation probability

TOURNSIZE = 5

# rest of experiment setup in BSE2 : main()
# eg. duration, other traders, ...
