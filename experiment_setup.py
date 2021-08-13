# general
END_TIME = 2500.0

# must be atleast 1
NUM_TRIALS = 1

# BUYERS_SPEC = [('ZIP', 50),('ZIC', 50),('GVWY', 50),('SHVR', 50)] 
# SELLERS_SPEC = [('ZIP', 50),('ZIC', 50),('GVWY', 50),('SHVR', 50)] 

# BUYERS_SPEC = [('ZIP', 20),('ZIC', 20),('GVWY', 20),('SHVR', 20)] 
# SELLERS_SPEC = [('ZIP', 20),('ZIC', 20),('GVWY', 20),('SHVR', 20)] 

# BUYERS_SPEC =  [('ZIC', 20),('ZIP', 20)] 
# SELLERS_SPEC = [('ZIC', 20),('ZIP', 20)] 

# BUYERS_SPEC =  [('ZIC', 150)] 
# SELLERS_SPEC = [('ZIC', 150)] 

# BUYERS_SPEC =  [('ZIP', 20)] 
# SELLERS_SPEC = [('ZIP', 20)] 


# BUYERS_SPEC =  [('ZIC', 300)] 
# SELLERS_SPEC = [('ZIC', 300)] 

BUYERS_SPEC =  [] 
SELLERS_SPEC = [] 

 


# STGP Entities
STGP_THERE = True

# only currently possible to have one of each
STGP_E_BUY = 1
STGP_E_SELL = 1

# STGP variables
# NUM_TRADERS_PER_ENTITY = 10
NUM_TRADERS_PER_ENTITY = 300
STGP_TRADER_STARTING_BALANCE = 100
NUM_GENS = 5
evolving = False

# probability 1 <= p <= 0
CXPB = 0.3 # crossover probability
MUTPB = 0.1 # mutation probability

TOURNSIZE = 3

# rest of experiment setup in BSE2 : main()
# eg. duration, other traders, ...
