# general
END_TIME = 1000.0
# must be atleast 1 - needs refactoring
BUYERS_SPEC = [('GVWY', 1)] 
SELLERS_SPEC = [('GVWY', 11)]

# STGP variables
NUM_TRADERS_PER_ENTITY = 10
STGP_TRADER_STARTING_BALANCE = 100
NUM_GENS = 10

# probability 1 <= p <= 0
CXPB = 0.5 # crossover probability
MUTPB = 0.2 # mutation probability

# rest of experiment setup in BSE2 : main()
# eg. duration, other traders, ...