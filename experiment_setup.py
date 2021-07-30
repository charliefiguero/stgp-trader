# general
END_TIME = 1000.0
# must be atleast 1 - needs refactoring
# BUYERS_SPEC = [('ZIP', 50),('ZIC', 50),('GVWY', 25),('SHVR', 25)] 
# SELLERS_SPEC = [('ZIP', 50),('ZIC', 50),('GVWY', 50),('SHVR', 25)] 

# BUYERS_SPEC =  [('ZIC', 20),('ZIP', 20)] 
# SELLERS_SPEC = [('ZIC', 20),('ZIP', 20)] 

# BUYERS_SPEC =  [('ZIC', 10)] 
# SELLERS_SPEC = [('ZIC', 10)] 

BUYERS_SPEC =  [('ZIP', 50)] 
SELLERS_SPEC = [('ZIP', 60)] 

# BUYERS_SPEC =  [] 
# SELLERS_SPEC = [] 

 


# STGP Entities
STGP_THERE = True

# only currently possible to have one of each
STGP_E_BUY = 1
STGP_E_SELL = 0

# STGP variables
NUM_TRADERS_PER_ENTITY = 10
STGP_TRADER_STARTING_BALANCE = 100
NUM_GENS = 10
evolving = True

# probability 1 <= p <= 0
CXPB = 0.3 # crossover probability
MUTPB = 0.1 # mutation probability

TOURNSIZE = 5

# rest of experiment setup in BSE2 : main()
# eg. duration, other traders, ...
