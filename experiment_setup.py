# general
END_TIME = 20000.0

# must be atleast 1
NUM_TRIALS = 1

# BUYERS_SPEC = [('ZIP', 200),('ZIC', 25),('GVWY', 25),('SHVR', 50)] 
# SELLERS_SPEC = [('ZIP', 200),('ZIC', 25),('GVWY', 25),('SHVR', 50)] 

# BUYERS_SPEC = [('ZIP', 20),('ZIC', 20),('GVWY', 20),('SHVR', 20)] 
# SELLERS_SPEC = [('ZIP', 20),('ZIC', 20),('GVWY', 20),('SHVR', 20)] 

# BUYERS_SPEC =  [('ZIC', 20),('ZIP', 20)] 
# SELLERS_SPEC = [('ZIC', 20),('', 20)] 

# BUYERS_SPEC =  [('ZIC', 150)] 
# SELLERS_SPEC = [('ZIC', 150)] 

# BUYERS_SPEC =  [('ZIP', 20)] 
# SELLERS_SPEC = [('ZIP', 20)] 


BUYERS_SPEC =  [('GVWY', 250)] 
SELLERS_SPEC = [('GVWY', 250)] 

# BUYERS_SPEC =  [] 
# SELLERS_SPEC = [] 

 


# STGP Entities
STGP_THERE = True


# only currently possible to have one of each
STGP_E_BUY = 1
STGP_E_SELL = 1

# STGP variables
# NUM_TRADERS_PER_ENTITY = 10
NUM_TRADERS_PER_ENTITY = 50
STGP_TRADER_STARTING_BALANCE = 100
NUM_GENS = 40
evolving = False

# probability 1 <= p <= 0
CXPB = 0.3 # crossover probability
MUTPB = 0.1 # mutation probability

TOURNSIZE = 3

# rest of experiment setup in BSE2 : main()
# eg. duration, other traders, ...
