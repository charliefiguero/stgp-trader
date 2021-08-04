import os
import glob
import csv
from datetime import datetime
import pickle
import ast
import statistics


import jsonpickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pygraphviz as pgv
from deap import gp


def read_pickle(fname):
    with open(fname, 'r') as file:
        data = file.read()
        return jsonpickle.decode(data)

def trader_gen_profits(trader_transactions):
    all_profits = []
    for gen in trader_transactions:
        gen_profits = sum([x.actual_improve for x in gen.transactions])
        all_profits.append(gen_profits)
    return all_profits

def gen_profits(all_traders):
    all_trader_profits = []
    for tr in all_traders:
        all_trader_profits.append(trader_gen_profits(tr))

    gen_profits = [sum(i) for i in zip(*all_trader_profits)]
    return pd.Series(gen_profits)

def plot_gen_profits(gen_profits, title = None):
    list_of_files = glob.glob('stgp_csvs/improvements/*') # * means all if need specific format then *.csv
    if not list_of_files:
        raise AssertionError('No records present. Try rerunning the experient.')

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f'reading file: ', latest_file, '\n')
    
    experiment_data = read_pickle(latest_file)
    exp_df = pd.DataFrame.from_dict(experiment_data)

    # calculate data
    gen_profits = gen_profits(exp_df['traders_data'])

    # plotting
    fig, ax = plt.subplots()
    ax.set_title('Generational Profits for STGP_Entity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Entity Profit')
    sns.set_theme()
    output = sns.lineplot(data=gen_profits, x=gen_profits.index, y=gen_profits, ax=ax)

    if title == None:
        title = datetime.now()
    output.get_figure().savefig(f'networth_plots/{title}.png')
    # plt.show()

def plot_stats():
    list_of_files = glob.glob('stgp_csvs/gen_records/*') # * means all if need specific format then *.csv
    if not list_of_files:
        raise AssertionError('No gen_records present. Try rerunning the experient.')

    latest_file = max(list_of_files, key=os.path.getctime)

    print(f'reading file: ', latest_file, '\n')

    experiment_data = read_pickle(latest_file)
    exp_df = pd.DataFrame.from_dict(experiment_data)
    exp_df.insert(0, 'gen_num', exp_df.index.tolist())
    exp_df['max'] = list(map(lambda x : x[0], exp_df['max']))
    exp_df['min'] = list(map(lambda x : x[0], exp_df['min']))

    print(exp_df)
    
    # PLOTTING

    # avg
    fig, ax = plt.subplots()
    ax.set_title('Average Profit for STGP_Traders')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Trader Profit')
    sns.set_theme()
    output = sns.lineplot(x=exp_df['gen_num'], y=exp_df['avg'], ax=ax)
    output.get_figure().savefig(f'stats_plots/avg/{datetime.now()}.png')

    # std
    fig, ax = plt.subplots()
    ax.set_title('Standard Deviation of Profit for STGP_Traders')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Standard Deviation')
    sns.set_theme()
    output = sns.lineplot(x=exp_df['gen_num'], y=exp_df['std'], ax=ax)
    output.get_figure().savefig(f'stats_plots/std/{datetime.now()}.png')

    # max
    fig, ax = plt.subplots()
    ax.set_title('Max Trader Profit for STGP')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Max Trader Profit')
    sns.set_theme()
    output = sns.lineplot(x=exp_df['gen_num'], y=exp_df['max'], ax=ax)
    output.get_figure().savefig(f'stats_plots/max/{datetime.now()}.png')

    # min
    fig, ax = plt.subplots()
    ax.set_title('Min Trader Profit for STGP')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Min Trader Profit')
    sns.set_theme()
    output = sns.lineplot(x=exp_df['gen_num'], y=exp_df['avg'], ax=ax)
    output.get_figure().savefig(f'stats_plots/min/{datetime.now()}.png')
    

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
    now = datetime.now()
    # print(f"Current time: {now}\n")
    if not name == None:
        g.draw(f"trees/tree {name}.pdf")
    else:
        g.draw(f"trees/tree {now}.pdf")

def plot_hof():
    list_of_files = glob.glob('stgp_csvs/hall_of_fame/*') # * means all if need specific format then *.csv
    if not list_of_files:
        raise AssertionError('No records present. Try rerunning the experient.')

    latest_file = max(list_of_files, key=os.path.getctime)

    with open(latest_file, 'rb') as infile:
        thawed_hof = pickle.load(infile)
        print(type(thawed_hof))
        print(thawed_hof)
        draw_expr(thawed_hof)

def mean_tran_price(duration: int, num_gens: int, fname: str):

    time_per_gen = duration/num_gens

    with open(fname, 'r') as infile:
        reader = csv.reader(infile)
        timeprice = [(row[2], row[3]) for row in reader]
        prices = [float(item[1]) for item in timeprice]

    print("Generational mean prices...")

    gen = 0
    trans_in_gen = 0
    gen_total_traded = 0
    for count, item in enumerate(timeprice):
        time = float(item[0])
        price = int(item[1])

        # gen reset
        while time > gen * time_per_gen:
            if gen == 0:
                gen += 1
                continue

            if trans_in_gen == 0:
                print(f'Gen: {gen}, mean price = {0}')
            else:
                print(f'Gen: {gen}, mean price = {gen_total_traded/trans_in_gen}')

            gen += 1
            trans_in_gen = 0
            gen_total_traded = 0

        gen_total_traded += price
        trans_in_gen += 1

    print(f'Average generational price: {sum(prices)/len(prices)}\n')



def blotter_debug():
    with open('Test00blotters.csv', 'r') as infile:
        reader = list(csv.reader(infile))
        # blot_b_lens = [x[1] for x in reader]
        blot_b_lens = [float(x[4]) for x in reader if x[1][1]=='B']
        blot_s_lens = [float(x[4]) for x in reader if x[1][1]=='S']

    from statistics import mean
    print(mean(blot_b_lens))
    print(mean(blot_s_lens))


def orders_prices():
    with open('Test00blotters.csv', 'r') as infile:
        reader = list(csv.reader(infile))
        b_blotters = [x[5:] for x in reader if x[1][1]=='B']
        s_blotters = [x[5:] for x in reader if x[1][1]=='S']

    b_orders = []
    for count in range(len(b_blotters)):
        t_bs = "".join(b_blotters[count]).split("|")
        b_orders.extend(t_bs)

    print(len(b_orders))

    s_orders = []
    for count in range(len(s_blotters)):
        t_bs = "".join(s_blotters[count]).split("|")
        s_orders.extend(t_bs)

    print(len(s_orders))

# def single_agent_efficiency(duration: int, num_gens: int, eq_price, filename):

#     with open(filename, 'r') as infile:
#         reader = list(csv.reader(infile))
    
#     trader_prices = {}
#     totals = []

#     # building a dictionary of every trade price for every trader
#     for tape in reader:
#         time = float(tape[2])
#         price = int(tape[3])

#         totals.append((time,price))

#         tran = tape[4:]
#         for i in range(len(tran)):
#             if i == len(tran)-1:
#                 break
#             tran[i] = tran[i] + ','
#         tran = ast.literal_eval("".join(tran)[1:])
        
#         seller = tran['party1']
#         buyer = tran['party2']

#         if seller in trader_prices:
#             trader_prices[seller].append((time, price))
#         else:
#             trader_prices[seller] = [(time, price)]

#         if buyer in trader_prices:
#             trader_prices[buyer].append((time, price))
#         else:
#             trader_prices[buyer] = [(time, price)]

#     # breaking trades into generations...

#     time_per_gen = duration/num_gens


#     # gen_time_prices[trader][generation]
#     gen_time_prices = {}

#     # iterate through traders in generation
#     for item in trader_prices.items():
#         trader_name = item[0]
#         trader_trans = item[1]
#         gen_time_prices[trader_name] = {}
#         gen = 0

#         # iterate through traders trades
#         for timeprice in trader_trans:
#             time = timeprice[0]
#             price = timeprice[1]

#             # trade takes place in next generation
#             if time > gen * time_per_gen:
#                 gen += 1
#                 gen_time_prices[trader_name][gen] = []

#             gen_time_prices[trader_name][gen].append(timeprice)

#     # check num_gens in function arguments match what is found
#     atrader = list(gen_time_prices.values())[0]
#     if len(atrader.keys()) != num_gens:
#         raise ValueError('number of gens or duration does not match experiment.'
#                         'num_gens in dictionary = %s', len(atrader.keys))


#     # Dict is now complete (trades per trader per generation). Now calculations...

#     stgp_mean_sea_per_gen = []
#     bstgp_mean_sea_per_gen = []
#     sstgp_mean_sea_per_gen = []
#     other_mean_sea_per_gen = []

#     bstgp_keys = [key for key in gen_time_prices.keys() if key.startswith('BSTGP')]
#     sstgp_keys = [key for key in gen_time_prices.keys() if key.startswith('SSTGP')]
#     stgp_keys = bstgp_keys + sstgp_keys
#     other_keys = [key for key in gen_time_prices.keys() if key not in stgp_keys]

#     print("Mean single agent efficiency per generation.")
#     for gen_num in range(num_gens):

#         stgp_sae = []
#         for t in stgp_keys:
#             try:
#                 t_trades = [x[1] for x in gen_time_prices[t][gen_num+1]]
#                 stgp_sae.append(sum(t_trades)/(len(t_trades)*100))
#             except:
#                 print(f"Trader {t} had no trades in generation {gen_num+1}.")

#         bstgp_sae = []
#         for t in bstgp_keys:
#             try:
#                 t_trades = [x[1] for x in gen_time_prices[t][gen_num+1]]
#                 bstgp_sae.append(sum(t_trades)/(len(t_trades)*100))
#             except:
#                 print(f"Trader {t} had no trades in generation {gen_num+1}.")

#         sstgp_sae = []
#         for t in sstgp_keys:
#             try:
#                 t_trades = [x[1] for x in gen_time_prices[t][gen_num+1]]
#                 sstgp_sae.append(sum(t_trades)/(len(t_trades)*100))
#             except:
#                 print(f"Trader {t} had no trades in generation {gen_num+1}.")

#         other_sae = []
#         for t in other_keys:
#             try:
#                 t_trades = [x[1] for x in gen_time_prices[t][gen_num+1]]
#                 other_sae.append(sum(t_trades)/(len(t_trades)*100))
#             except:
#                 print(f"Trader {t} had no trades in generation {gen_num+1}.")

#         stgp_mean_sae = statistics.mean(stgp_sae)
#         bstgp_mean_sae = statistics.mean(bstgp_sae)
#         sstgp_mean_sae = statistics.mean(sstgp_sae)
#         other_mean_sae = statistics.mean(other_sae)

#         stgp_mean_sea_per_gen.append(stgp_mean_sae)
#         bstgp_mean_sea_per_gen.append(bstgp_mean_sae)
#         sstgp_mean_sea_per_gen.append(sstgp_mean_sae)
#         other_mean_sea_per_gen.append(other_mean_sae)
        
#         # print(f"Gen {gen_num+1}: BSTGP mean = {bstgp_mean}, Others mean = {other_mean}")
#         print('Gen: {0:<4}, STGP = {1:<20}, BSTGP = {2:<20}, SSTGP = {3:<20}, Others = {4:<15}'
#                     .format(gen_num+1,stgp_mean_sae, bstgp_mean_sae, sstgp_mean_sae, other_mean_sae))

def sae(duration: int, num_gens: int, eq_price, filename):
    with open(filename, 'r') as infile:
        reader = list(csv.reader(infile))

    time_per_gen = duration / num_gens
    eq_price = 100

    trader_profit = {}

    # 1 trader per row
    for row in reader:
        tid = row[1]
        ttype = row[2]
        num_trades = row[3]
        trades = [list(map(float, x[1:].split(' '))) for x in row[4:]]

        trader_profit[tid] = {}

        gen = 0
        for t in trades:
            time = t[0]
            profit = t[1]
            cprice = t[2]
            tradeprice = t[3]

            if "B" in tid:
                expectedprofit = cprice - eq_price
            else:
                expectedprofit = eq_price - cprice

            while time > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []
            
            trader_profit[tid][gen].append((profit, expectedprofit))

        # create generation arrays for generations with no trades
        if len(trader_profit[tid]) < num_gens:
            while duration > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []

    BSTGP_keys = [x for x in trader_profit.keys() if "BSTGP" in x]
    SSTGP_keys = [x for x in trader_profit.keys() if "SSTGP" in x]
    BOTHER_keys = [x for x in trader_profit.keys() if "B" in x and x not in BSTGP_keys]
    SOTHER_keys = [x for x in trader_profit.keys() if "S" in x and x not in SSTGP_keys]

    def group_gen_sae(keys, gen):
        actualprofits = 0
        expectedprofits = 0
        for tkey in keys:
            trader_gen = trader_profit[tkey][gen]
            for trade in trader_gen:
                actualprofits += trade[0]
                expectedprofits += trade[1]

        return actualprofits / expectedprofits

    print("Generational SAE...")
    for gen in range(1, num_gens+1):
        
        BSTGP_sae = group_gen_sae(BSTGP_keys, gen)
        SSTGP_sae = group_gen_sae(SSTGP_keys, gen)
        BOTHER_sae = group_gen_sae(BOTHER_keys, gen)
        SOTHER_sae = group_gen_sae(SOTHER_keys, gen)

        print(f"Gen {gen:<10}: BSTGP={BSTGP_sae:<20}, SSTGP={SSTGP_sae:<20}, BOTHER={BOTHER_sae:<20}, SOTHER={SOTHER_sae:<20}")
    

def sae_series():
    num_gens = 10
    duration = 5000
    eq_price = 100

    num_trials = 10

    tape_files = [f for f in os.listdir('./standard_csvs') if 'Test' in f and 'tapes' in f]
    print(tape_files)

    gen_sae_bins = {}
    for i in range(num_trials):
        gen_sae_bins[i+1] = []

    for tape in tape_files:
        print(tape)
        fpath = 'standard_csvs/'+tape
        trial_sae = single_agent_efficiency(duration, num_gens, eq_price, fpath)

        # looping over the sae for the gens in the trial
        for index, gen in enumerate(trial_sae):
            gen_sae_bins[index+1].append(gen)

    # print final bin counts
    print(f'\n\nAverage SAE per gen over {len(tape_files)} trials.\n')
    for item in gen_sae_bins.items():
        print(item[0], statistics.mean(item[1]))

def plot_tran_price():

    def line_plot(x, y):
        _, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_ylim(ymin=0)
        return ax

    def plot_transactions(times, prices, title=None):
        ax = line_plot(times, prices)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Price (£)')
        if title != None:
            ax.set_title(title)
        return ax

    with open('Test00tapes.csv', 'r') as infile:
        reader = csv.reader(infile)
        timeprice = [(row[2], row[3]) for row in reader]
        times = [float(item[0]) for item in timeprice]
        prices = [float(item[1]) for item in timeprice]

    ax = plot_transactions(times, prices)
    plt.show()



if __name__ == "__main__":
    # plot_stats()

    num_gens = 10
    duration = 5000
    eq_price = 100
    num_trials = 1
    fpath = "standard_csvs/Test00tapes.csv"

    # mean_tran_price(duration, num_gens, fpath)
    # single_agent_efficiency(duration, num_gens, eq_price, fpath)

    sae(500, 10, 100, "standard_csvs/Test00profit.csv")

        
    # orders_prices()
    # plot_tran_price()
    # blotter_debug()
    # plot_hof()

    # num_gens = 1
    # duration = 10000
    # answer = single_agent_efficiency(duration/num_gens)
    # print(answer)



    
    # list_of_files = glob.glob('stgp_csvs/gen_records/*') # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # print(f'reading file: ', latest_file, '\n')

    # experiment_data = read_pickle(latest_file)
    # exp_df = pd.DataFrame.from_dict(experiment_data)
    # exp_df.insert(0, 'gen_num', exp_df.index.tolist())
    # exp_df['max'] = list(map(lambda x : x[0], exp_df['max']))
    # exp_df['min'] = list(map(lambda x : x[0], exp_df['min']))

    # print(exp_df)

    # exp_df.plot(x='gen_num', y='avg', kind='line')
    # plt.show()

    # exp_df.plot(x='gen_num', y='std', kind='line')
    # plt.show()

    # exp_df.plot(x='gen_num', y='max', kind='line')
    # plt.show()

    # exp_df.plot(x='gen_num', y='min', kind='line')
    # plt.show()




    # list_of_files = glob.glob('stgp_csvs/improvements/*') # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # print(f'reading file: ', latest_file, '\n')
    
    # experiment_data = read_pickle(latest_file)
    # exp_df = pd.DataFrame.from_dict(experiment_data)


    # # calculate data
    # gen_profits = gen_profits(exp_df['traders_data'])

    # # plotting
    # print(gen_profits)
    # plot_gen_profits(gen_profits)





    # print(traders_gen_profits(exp_df['traders_data'][0]))
    # print(gen_profits(exp_df['traders_data']))