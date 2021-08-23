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
import numpy as np


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


def plotstats_one_entity(fname):

    bfile = fname

    print(f'reading file: ', bfile, '\n')

    bexperiment_data = read_pickle(bfile)
    bexp_df = pd.DataFrame.from_dict(bexperiment_data)
    bexp_df.insert(0, 'gen_num', bexp_df.index.tolist())
    bexp_df['max'] = list(map(lambda x : x[0], bexp_df['max']))
    bexp_df['min'] = list(map(lambda x : x[0], bexp_df['min']))

    print(bexp_df)


    _, ax = plt.subplots()

    x = list(map(lambda x: x+1, bexp_df['gen_num'].tolist()))

    ax.set_title('Profit for GP Buyers')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Trader Profit')

    
    yavg = bexp_df['avg'].tolist()
    ymax = bexp_df['max'].tolist()
    err = bexp_df['std'].tolist()
    ax.errorbar(x=x, y=yavg, fmt="ko", linewidth=1, linestyle="-", yerr=err, capsize=1, elinewidth = 0.5)
    ax.plot(x, ymax, "ro", linewidth=1, linestyle="--")

    plt.show()

def plot_stats(bfname, sfname):
    # list_of_files = glob.glob('stgp_csvs/gen_records/*') # * means all if need specific format then *.csv
    # if not list_of_files:
    #     raise AssertionError('No gen_records present. Try rerunning the experient.')

    # latest_file = max(list_of_files, key=os.path.getctime)

    # #####   BUYERS   ########
    bfile = bfname

    print(f'reading file: ', bfile, '\n')

    bexperiment_data = read_pickle(bfile)
    bexp_df = pd.DataFrame.from_dict(bexperiment_data)
    bexp_df.insert(0, 'gen_num', bexp_df.index.tolist())
    bexp_df['max'] = list(map(lambda x : x[0], bexp_df['max']))
    bexp_df['min'] = list(map(lambda x : x[0], bexp_df['min']))

    print(bexp_df)


    fig, (ax1, ax2) = plt.subplots(1,2)

    x = list(map(lambda x: x+1, bexp_df['gen_num'].tolist()))

    ax1.set_title('Profit for GP Buyers')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Average Trader Profit')

    
    yavg = bexp_df['avg'].tolist()
    ymax = bexp_df['max'].tolist()
    err = bexp_df['std'].tolist()
    ax1.errorbar(x=x, y=yavg, fmt="ko", linewidth=1, linestyle="-", yerr=err, capsize=1, elinewidth = 0.5)
    ax1.plot(x, ymax, "ro", linewidth=1, linestyle="--")

    


    # ########## SELLERS ###########

    sfile = sfname

    print(f'reading file: ', sfile, '\n')

    sexperiment_data = read_pickle(sfile)
    sexp_df = pd.DataFrame.from_dict(sexperiment_data)
    sexp_df.insert(0, 'gen_num', sexp_df.index.tolist())
    sexp_df['max'] = list(map(lambda x : x[0], sexp_df['max']))
    sexp_df['min'] = list(map(lambda x : x[0], sexp_df['min']))

    print(sexp_df)

    
    ax2.set_title('Profit for GP Sellers')
    ax2.set_xlabel('Generation')
    # ax2.set_ylabel('Average Trader Profit')

    
    yavg = sexp_df['avg'].tolist()
    ymax = sexp_df['max'].tolist()
    err = sexp_df['std'].tolist()
    ax2.errorbar(x=x, y=yavg, fmt="ko", linewidth=1, linestyle="-", yerr=err, capsize=1, elinewidth = 0.5)
    ax2.plot(x, ymax, "ro", linewidth=1, linestyle="--")


    plt.show()

    

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
    gen_mean_price = []
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
                gen_mean_price.append(0)
            else:
                print(f'Gen: {gen}, mean price = {gen_total_traded/trans_in_gen}')
                gen_mean_price.append(gen_total_traded/trans_in_gen)

            gen += 1
            trans_in_gen = 0
            gen_total_traded = 0

        gen_total_traded += price
        trans_in_gen += 1

    if trans_in_gen == 0:
        print(f'Gen: {gen}, mean price = {0}')
        gen_mean_price.append(0)
    else:
        print(f'Gen: {gen}, mean price = {gen_total_traded/trans_in_gen}')
        gen_mean_price.append(gen_total_traded/trans_in_gen)

    print(f'Average generational price: {sum(prices)/len(prices)}\n')
    return gen_mean_price


def plot_gen_mean_price(mean_prices, eq_price):
    _, ax = plt.subplots()
    plt.xlabel("Generation")
    plt.ylabel("Mean Price")
    ax.set_title("Mean Price: GP vs GP")

    gens_range = list(map(lambda x: x+1, range(len(mean_prices))))
    eq_price_line = [eq_price] * len(mean_prices)
    ax.plot(gens_range, eq_price_line, ":k")
    ax.plot(gens_range, mean_prices, "-b")
    plt.show()


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

def genprofit(duration: int, num_gens: int, eq_price, filename):

    with open(filename, 'r') as infile:
        reader = list(csv.reader(infile))

    time_per_gen = duration / num_gens
    eq_price = 100

    trader_profit = {}

    # 1 trader per row
    for row in reader:
        tid = row[1][1:]
        ttype = row[2][1:]
        num_trades = row[3]
        trades = [list(map(float, x[1:].split(' '))) for x in row[4:]]

        trader_profit[tid] = {}
        trader_profit[tid]['ttype'] = ttype

        gen = 0
        for t in trades:
            time = t[0]
            profit = t[1]

            while time > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []
            
            trader_profit[tid][gen].append(profit)

        # create generation arrays for generations with no trades
        if len(trader_profit[tid]) < num_gens+1:
            while duration > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []

    BSTGP_keys = [x for x in trader_profit.keys() if "BSTGP" in x]
    SSTGP_keys = [x for x in trader_profit.keys() if "SSTGP" in x]
    BOTHER_keys = [x for x in trader_profit.keys() if "B" in x and x not in BSTGP_keys]
    SOTHER_keys = [x for x in trader_profit.keys() if "S" in x and x not in SSTGP_keys and x not in BSTGP_keys]

    def group_gen_profit(keys, gen):
        total_profit = 0
        for tkey in keys:
            total_profit += sum(trader_profit[tkey][gen])

        if len(keys) == 0:
            return 0
        else:
            return total_profit / len(keys)

    BSTGP_gen_profit  = []
    SSTGP_gen_profit  = []
    BOTHER_gen_profit = []
    SOTHER_gen_profit = []

    print("Generational Profit...")
    for gen in range(1, num_gens+1):
        
        BSTGP_profit = group_gen_profit(BSTGP_keys, gen)
        SSTGP_profit = group_gen_profit(SSTGP_keys, gen)
        BOTHER_profit = group_gen_profit(BOTHER_keys, gen)
        SOTHER_profit = group_gen_profit(SOTHER_keys, gen)

        BSTGP_gen_profit.append(BSTGP_profit) 
        SSTGP_gen_profit.append(SSTGP_profit) 
        BOTHER_gen_profit.append(BOTHER_profit)
        SOTHER_gen_profit.append(SOTHER_profit)

        print(f"Gen {gen:<10}: BSTGP={BSTGP_profit:<20}, SSTGP={SSTGP_profit:<20}, BOTHER={BOTHER_profit:<20}, SOTHER={SOTHER_profit:<20}")

    return BSTGP_gen_profit, SSTGP_gen_profit, BOTHER_gen_profit, SOTHER_gen_profit


def numtrades(duration: int, num_gens: int, eq_price, filename):

    with open(filename, 'r') as infile:
        reader = list(csv.reader(infile))

    time_per_gen = duration / num_gens
    eq_price = 100

    trader_profit = {}

    # 1 trader per row
    for row in reader:
        tid = row[1][1:]
        ttype = row[2][1:]
        num_trades = row[3]
        # print(row[4:])

        # print('\n\n\n\n\n')
        trades = [list(map(float, x[1:].split(' '))) for x in row[4:]]

        trader_profit[tid] = {}
        trader_profit[tid]['ttype'] = ttype

        gen = 0
        # print(row)
        for t in trades:
            # print(t)
            time = t[0]
            profit = t[1]

            while time > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []
            
            trader_profit[tid][gen].append(profit)

        # create generation arrays for generations with no trades
        if len(trader_profit[tid]) < num_gens+1:
            while duration > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []

    BSTGP_keys = [x for x in trader_profit.keys() if "BSTGP" in x]
    SSTGP_keys = [x for x in trader_profit.keys() if "SSTGP" in x]
    BOTHER_keys = [x for x in trader_profit.keys() if "B" in x and x not in BSTGP_keys]
    SOTHER_keys = [x for x in trader_profit.keys() if "S" in x and x not in SSTGP_keys and x not in BSTGP_keys]

    def group_gen_profit(keys, gen):
        total_profit = 0
        for tkey in keys:
            total_profit += sum(trader_profit[tkey][gen])
        return total_profit / len(keys)

    def get_num_trades(keys, gen):
        num = 0
        for tkey in keys:
            num += len(trader_profit[tkey][gen])
        if len(keys) == 0:
            return 0
        else:
            return num / len(keys)

    BSTGP_gen_num_trades  = []
    SSTGP_gen_num_trades  = []
    BOTHER_gen_num_trades = []
    SOTHER_gen_num_trades = []

    print("Generational Num Trades...")
    for gen in range(1, num_gens+1):

        BSTGP_num_trades = get_num_trades(BSTGP_keys, gen)
        SSTGP_num_trades = get_num_trades(SSTGP_keys, gen)
        BOTHER_num_trades = get_num_trades(BOTHER_keys, gen)
        SOTHER_num_trades = get_num_trades(SOTHER_keys, gen)

        BSTGP_gen_num_trades.append(BSTGP_num_trades) 
        SSTGP_gen_num_trades.append(SSTGP_num_trades) 
        BOTHER_gen_num_trades.append(BOTHER_num_trades)
        SOTHER_gen_num_trades.append(SOTHER_num_trades)

        print(f"Gen {gen:<10}: BSTGP={BSTGP_num_trades:<20}, SSTGP={SSTGP_num_trades:<20}, BOTHER={BOTHER_num_trades:<20}, SOTHER={SOTHER_num_trades:<20}")

    return BSTGP_gen_num_trades, SSTGP_gen_num_trades, BOTHER_gen_num_trades, SOTHER_gen_num_trades


def sae(duration: int, num_gens: int, eq_price, filename):
    with open(filename, 'r') as infile:
        reader = list(csv.reader(infile))

    time_per_gen = duration / num_gens
    eq_price = 100

    trader_profit = {}

    # 1 trader per row
    for row in reader:
        tid = row[1][1:]
        ttype = row[2][1:]
        num_trades = row[3]
        trades = [list(map(float, x[1:].split(' '))) for x in row[4:]]

        trader_profit[tid] = {}
        trader_profit[tid]['ttype'] = ttype

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
        if len(trader_profit[tid]) < num_gens + 1: # ttype is a key
            while duration > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []

    BSTGP_keys = [x for x in trader_profit.keys() if "BSTGP" in x]
    SSTGP_keys = [x for x in trader_profit.keys() if "SSTGP" in x]
    BOTHER_keys = [x for x in trader_profit.keys() if "B" in x and x not in BSTGP_keys]
    SOTHER_keys = [x for x in trader_profit.keys() if "S" in x and x not in SSTGP_keys and x not in BSTGP_keys]


    def group_gen_sae(keys, gen):
        actualprofits = 0
        expectedprofits = 0
        for tkey in keys:
            trader_gen = trader_profit[tkey][gen]
            for trade in trader_gen:
                actualprofits += trade[0]
                expectedprofits += trade[1]

        return (actualprofits + 1) / (expectedprofits + 1)

    BSTGP_gen_sae  = []
    SSTGP_gen_sae  = []
    BOTHER_gen_sae = []
    SOTHER_gen_sae = []

    print("Generational SAE...")
    for gen in range(1, num_gens+1):
        
        BSTGP_sae = group_gen_sae(BSTGP_keys, gen)
        SSTGP_sae = group_gen_sae(SSTGP_keys, gen)
        BOTHER_sae = group_gen_sae(BOTHER_keys, gen)
        SOTHER_sae = group_gen_sae(SOTHER_keys, gen)

        BSTGP_gen_sae.append(BSTGP_sae) 
        SSTGP_gen_sae.append(SSTGP_sae) 
        BOTHER_gen_sae.append(BOTHER_sae)
        SOTHER_gen_sae.append(SOTHER_sae)

        print(f"Gen {gen:<10}: BSTGP={BSTGP_sae:<20}, SSTGP={SSTGP_sae:<20}, BOTHER={BOTHER_sae:<20}, SOTHER={SOTHER_sae:<20}")

    return BSTGP_gen_sae, SSTGP_gen_sae, BOTHER_gen_sae, SOTHER_gen_sae

    
    

def sae_series():
    num_gens = 20
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
        ax.set_title("Transaction Price")
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

    with open('standard_csvs/Test00tapes.csv', 'r') as infile:
        reader = csv.reader(infile)
        timeprice = [(row[2], row[3]) for row in reader]
        times = [float(item[0]) for item in timeprice]
        prices = [float(item[1]) for item in timeprice]

    ax = plot_transactions(times, prices)
    plt.show()

def plot_gen_sae(BSTGP_gen_sae, SSTGP_gen_sae, BOTHER_gen_sae, SOTHER_gen_sae, mean_profits):

    def line_plot(ax, x, y):
        ax.plot(x, y)
        ax.set_ylim(ymin=0)

    def plot_transactions(times, prices, title=None):
        ax = line_plot(times, prices)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Trader SAE')
        if title != None:
            ax.set_title(title)
        return ax

    _, ax = plt.subplots()
    plt.xlabel("Generation")
    plt.ylabel("Single Agent Efficiency")
    ax.set_title("SAE")

    gens_range = range(1, num_gens+1)
    expected_sae = [1]*num_gens
    ax.plot(gens_range, expected_sae, ":k")
    ax.plot(gens_range, BSTGP_gen_sae, "-g", label="GP Buyers")
    ax.plot(gens_range, SSTGP_gen_sae, "-r", label="GP Sellers")
    # ax.plot(gens_range, mean_profits, "-c", linewidth=2.0, label="Mean Price")
    ax.plot(gens_range, BOTHER_gen_sae, "--g", label="ZIC Buyers")
    ax.plot(gens_range, SOTHER_gen_sae, "--r", label="ZIC Sellers")
    plt.legend()
    plt.show()


def plot_gen_profit(BSTGP_gen_profit, SSTGP_gen_profit, BOTHER_gen_profit, SOTHER_gen_profit, mean_profits):

    def line_plot(ax, x, y):
        ax.plot(x, y)
        ax.set_ylim(ymin=0)

    def plot_transactions(times, prices, title=None):
        ax = line_plot(times, prices)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Trader Profit')
        if title != None:
            ax.set_title(title)
        return ax

    _, ax = plt.subplots()
    ax.set_title("Profit")
    plt.xlabel("Generation")
    plt.ylabel("Mean Profit")

    gens_range = range(1, num_gens+1)

    ax.plot(gens_range, BSTGP_gen_profit, "-g", label="GP Buyers")
    ax.plot(gens_range, SSTGP_gen_profit, "-r", label="GP Sellers")
    # ax.plot(gens_range, mean_profits, "-c", linewidth=2.0, label="Mean Price")
    ax.plot(gens_range, BOTHER_gen_profit, "--g", label="ZIC Buyers")
    ax.plot(gens_range, SOTHER_gen_profit, "--r", label="ZIC Sellers")
    plt.legend()
    plt.show()

def plot_gen_numtrades(BSTGP_gen_num_trades, SSTGP_gen_num_trades, BOTHER_gen_num_trades, SOTHER_gen_num_trades, mean_prices):

    def line_plot(ax, x, y):
        ax.plot(x, y)
        ax.set_ylim(ymin=0)

    def plot_transactions(times, prices, title=None):
        ax = line_plot(times, prices)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Trader Profit')
        if title != None:
            ax.set_title(title)
        return ax

    _, ax = plt.subplots()
    ax.set_title("Number of Trades")
    plt.xlabel("Generation")
    plt.ylabel("Number of Trades")

    gens_range = range(1, num_gens+1)

    ax.plot(gens_range, BSTGP_gen_num_trades, "-g", label="GP Buyers")
    ax.plot(gens_range, SSTGP_gen_num_trades, "-r", label="GP Sellers")
    # ax.plot(gens_range, mean_prices, "--c", label="Mean Prices")
    ax.plot(gens_range, BOTHER_gen_num_trades, "--g", label="ZIC Buyers")
    ax.plot(gens_range, SOTHER_gen_num_trades, "--r", label="ZIC Sellers")
    plt.legend()
    plt.show()

def quoteprice_analysis(duration, num_gens, filename):
    with open(filename, 'r') as infile:
        reader = list(csv.reader(infile))

    time_per_gen = duration / num_gens

    trader_quotes = {}

    # 1 trader per row
    for row in reader:
        tid = row[1][1:]
        # print(tid)
        num_trades = row[2]
        quotes = [list(map(float, x[1:].split(' '))) for x in row[3:]]

        trader_quotes[tid] = {}

        gen = 0
        for t in quotes:
            time = t[0]
            quoteprice = t[1]

            while time > gen * time_per_gen: 
                gen += 1
                trader_quotes[tid][gen] = []
            
            trader_quotes[tid][gen].append(quoteprice)

        # create generation arrays for generations with no trades
        if len(trader_quotes[tid]) < num_gens:
            while duration > gen * time_per_gen: 
                gen += 1
                trader_quotes[tid][gen] = []

    BSTGP_keys = [x for x in trader_quotes.keys() if "BSTGP" in x]
    SSTGP_keys = [x for x in trader_quotes.keys() if "SSTGP" in x]
    BOTHER_keys = [x for x in trader_quotes.keys() if "B" in x and x not in BSTGP_keys]
    SOTHER_keys = [x for x in trader_quotes.keys() if "S" in x and x not in SSTGP_keys and x not in BSTGP_keys]

    def group_gen_average_quote(keys, gen):
        total = 0
        count = 0
        for tkey in keys:
            trader_gen = trader_quotes[tkey][gen]
            total += sum(trader_gen)
            count += len(trader_gen)

        if count < 1:
            return 0
        else:
            return (total) / (count)

    BSTGP_gen_meanquote  = []
    SSTGP_gen_meanquote  = []
    BOTHER_gen_meanquote = []
    SOTHER_gen_meanquote = []

    print("Generational Mean Quote Price...")
    for gen in range(1, num_gens+1):
        
        BSTGP_meanquote = group_gen_average_quote(BSTGP_keys, gen)
        SSTGP_meanquote = group_gen_average_quote(SSTGP_keys, gen)
        BOTHER_meanquote = group_gen_average_quote(BOTHER_keys, gen)
        SOTHER_meanquote = group_gen_average_quote(SOTHER_keys, gen)

        BSTGP_gen_meanquote.append(BSTGP_meanquote) 
        SSTGP_gen_meanquote.append(SSTGP_meanquote) 
        BOTHER_gen_meanquote.append(BOTHER_meanquote)
        SOTHER_gen_meanquote.append(SOTHER_meanquote)

        print(f"Gen {gen:<10}: BSTGP={BSTGP_meanquote:<20}, SSTGP={SSTGP_meanquote:<20}, BOTHER={BOTHER_meanquote:<20}, SOTHER={SOTHER_meanquote:<20}")

    return BSTGP_gen_meanquote, SSTGP_gen_meanquote, BOTHER_gen_meanquote, SOTHER_gen_meanquote


def plot_gen_meanquote(BSTGP_gen_meanquote, SSTGP_gen_meanquote, BOTHER_gen_meanquote, SOTHER_gen_meanquote, mean_prices, eq_price):

    def line_plot(ax, x, y):
        ax.plot(x, y)
        ax.set_ylim(ymin=0)

    def plot_transactions(times, prices, title=None):
        ax = line_plot(times, prices)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Trader Profit')
        if title != None:
            ax.set_title(title)
        return ax

    _, ax = plt.subplots()
    ax.set_title("Mean and Quote Prices")
    plt.xlabel("Generation")
    plt.ylabel("Price")

    gens_range = range(1, num_gens+1)
    eq_prices = [eq_price] * len(gens_range)

    ax.plot(gens_range, BSTGP_gen_meanquote, "-g", label="GP Buyers")
    ax.plot(gens_range, SSTGP_gen_meanquote, "-r", label="GP Sellers")
    ax.plot(gens_range, eq_prices, ":k", label="Equilibrium")
    ax.plot(gens_range, mean_prices, ":c", label="Mean Price")
    ax.plot(gens_range, BOTHER_gen_meanquote, "--g", label="ZIC Buyers")
    ax.plot(gens_range, SOTHER_gen_meanquote, "--r", label="ZIC Sellers")
    plt.legend()
    plt.show()




# def plot_gen_mean_price_two_prices(meanprices1, meanprices2, eq_price):
#     _, ax = plt.subplots()
#     plt.xlabel("Generation")
#     plt.ylabel("Mean Price")
#     ax.set_title("Mean Price: GP vs GP")

#     gens_range = list(map(lambda x: x+1, range(len(mean_prices))))
#     eq_price_line = [eq_price] * len(mean_prices)


#     ax.plot(gens_range, eq_price_line, ":k")
#     ax.plot(gens_range, meanprices1, "-b", label="With GP")
#     ax.plot(gens_range, meanprices2, "-m", label="Without GP")
#     plt.legend()
#     plt.show()



if __name__ == "__main__":
    # plot_stats()

    num_gens = 40
    duration = 20000
    eq_price = 100
    num_trials = 1
    # fpath = "standard_csvs/Test00tapes.csv"
    # profit_fpath = "standard_csvs/Test00profit.csv"




    fpath = "experiments/zic_evol/Test00tapes.csv"
    profit_fpath = "standard_csvs/Test00profit.csv"


    # mean_prices = mean_tran_price(duration, num_gens, fpath)
    # # plot_gen_mean_price(mean_prices, eq_price)

    # BSTGP_gen_sae, SSTGP_gen_sae, BOTHER_gen_sae, SOTHER_gen_sae = sae(duration, num_gens, eq_price, profit_fpath)
    # plot_gen_sae(BSTGP_gen_sae, SSTGP_gen_sae, BOTHER_gen_sae, SOTHER_gen_sae, mean_prices)
    
    BSTGP_gen_profit, SSTGP_gen_profit, BOTHER_gen_profit, SOTHER_gen_profit = genprofit(duration, num_gens, eq_price, profit_fpath) 
    # plot_gen_profit(BSTGP_gen_profit, SSTGP_gen_profit, BOTHER_gen_profit, SOTHER_gen_profit, mean_prices)


    # Average profit:
    print("GP: " + str(statistics.mean(BSTGP_gen_profit + SSTGP_gen_profit)))
    print("OTHER: " + str(statistics.mean(BOTHER_gen_profit + SOTHER_gen_profit)))


    # BSTGP_gen_num_trades, SSTGP_gen_num_trades, BOTHER_gen_num_trades, SOTHER_gen_num_trades = numtrades(duration, num_gens, eq_price, profit_fpath)
    # plot_gen_numtrades(BSTGP_gen_num_trades, SSTGP_gen_num_trades, BOTHER_gen_num_trades, SOTHER_gen_num_trades, mean_prices)

    # quote_fpath="experiments/zic_evol/Test00quotes.csv"
    # BSTGP_gen_meanquote, SSTGP_gen_meanquote, BOTHER_gen_meanquote, SOTHER_gen_meanquote = quoteprice_analysis(duration, num_gens, quote_fpath)

    # plot_gen_meanquote(BSTGP_gen_meanquote, SSTGP_gen_meanquote, BOTHER_gen_meanquote, SOTHER_gen_meanquote, mean_prices, eq_price)

    # sgpfile = "experiments/zic_evol/2021-08-19 07:23:10.447009S_STGP_ENTITY_0.json"
    # bgpfile = "experiments/zic_evol/2021-08-19 07:23:05.032957B_STGP_ENTITY_0.json"



    # plot_stats(bgpfile, sgpfile)
    # plotstats_one_entity("experiments/oneup_run.json")

    exit()





    # compiling mean difference between gp and zip sae each generation

    bsae_diff = {}
    ssae_diff = {}

    bsae = {}
    ssae = {}
    bothersae = {}
    sothersae = {}


    for i in range(10):

        print(f"Session: {i}")

        path = "experiments/comp_newzip/"
        fname = path+f"Test{i:02}profit.csv"
        # fname = path+f"Test{4:02}profit.csv"

        BSTGP_gen_sae, SSTGP_gen_sae, BOTHER_gen_sae, SOTHER_gen_sae = genprofit(duration, num_gens, eq_price, fname)


        for i in range(num_gens):
            gen = i+1

            # bdiff = BSTGP_gen_sae[i]
            bdiff = BSTGP_gen_sae[i] - BOTHER_gen_sae[i]
            if gen in bsae_diff:
                bsae_diff[gen] += bdiff
            else:
                bsae_diff[gen] = bdiff

            # sdiff = SSTGP_gen_sae[i]
            sdiff = SSTGP_gen_sae[i] - SOTHER_gen_sae[i]
            if gen in ssae_diff:
                ssae_diff[gen] += sdiff
            else:
                ssae_diff[gen] = sdiff





            b = BSTGP_gen_sae[i]
            if gen in bsae:
                bsae[gen] += b
            else:
                bsae[gen] = b

            s = SSTGP_gen_sae[i]
            if gen in ssae:
                ssae[gen] += s
            else:
                ssae[gen] = s

            bother = BOTHER_gen_sae[i]
            if gen in bothersae:
                bothersae[gen] += bother
            else:
                bothersae[gen] = bother

            sother = SOTHER_gen_sae[i]
            if gen in sothersae:
                sothersae[gen] += sother
            else:
                sothersae[gen] = sother


            

    

    # 10 for the number of trials
    bsae_diff = list(map(lambda x: x/(10), bsae_diff.values()))
    ssae_diff = list(map(lambda x: x/(10), ssae_diff.values()))

    
    bsae = list(map(lambda x: x/(10), bsae.values()))
    ssae = list(map(lambda x: x/(10), ssae.values()))
    bothersae = list(map(lambda x: x/(10), bothersae.values()))
    sothersae = list(map(lambda x: x/(10), sothersae.values()))


    combined = []
    for i in range(len(bsae_diff)):
        combined.append((bsae_diff[i]+ssae_diff[i])/2)

    
    # print(combined)
    # print(combined[1:])

    gp = [(a+b)/2 for a, b in zip(bsae, ssae)]
    zipper = [(a+b)/2 for a, b in zip(bothersae, sothersae)]
    


    _, ax = plt.subplots()
    ax.set_title("Average Generational Profit: GP and ZIP")
    plt.xlabel("Generation")
    plt.ylabel("Profit")

    gens_range = range(1, num_gens+1)
    eq_prices = [1] * len(gens_range)

    m, c = np.polyfit(gens_range[2:], gp[2:], 1)
    print(f"buy: m={m}, c={c}")

    # ax.plot(gens_range, bsae_diff, "-g", label="Buyers")
    # ax.plot(gens_range, ssae_diff, "-r", label="Sellers")
    # ax.plot(gens_range, combined, "-b")


    ax.plot(gens_range, gp, "-b", label="GP Traders")
    ax.plot(gens_range, zipper, "-m", label="ZIP Traders")

    # ax.plot(gens_range, bsae, "-g")
    # ax.plot(gens_range, ssae, "-r")
    # ax.plot(gens_range, bothersae, "--g")
    # ax.plot(gens_range, sothersae, "--r")


    # ax.plot(gens_range, eq_prices, ":k")
    ax.plot(gens_range[2:], gens_range[2:]*m + c, ":c")


    # ax.plot(gens_range, eq_prices, ":k", label="Equilibrium")
    plt.legend()
    plt.show()


    # by = list(map(lambda x: x/10, bgenerational_bins.values()))
    # bm, bc = np.polyfit(gens_range, combined, 1)
    # print(f"buy: m={bm}, c={bc}")

    # # sell side
    # sy = list(map(lambda x: x/10, sgenerational_bins.values()))
    # sm, sc = np.polyfit(gens_range, sy, 1)
    # print(f"buy: m={sm}, c={sc}")









        
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




    # numgens = 40
    # gens_range = range(1, numgens+1)
    # eq_prices = [100] * len(gens_range)
    
    # # buy side
    # bgenerational_bins = {}
    # # sell side
    # sgenerational_bins = {}

    # # _, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(3,3)

  
    # x=0
    # y=0
    # for i in range(10):
    #     print(f"Session: {i}")


    #     path = "experiments/comp_newzip/"
    #     fname = path+f"Test{i:02}profit.csv"

    #     bgp, sgp, bother, sother = sae(duration, numgens, 100, filename=fname)
    #     # print(bgp[0], sgp[0], bother[0], sother[0])
    #     # print(statistics.mean(bgp))
    #     # print(statistics.mean(sgp))
    #     # print(statistics.mean(bother))
    #     # print(statistics.mean(sother))
    #     # exit()
    #     # bgp, sgp, bother, sother = sae(duration, numgens, 100, filename=fname)



    #     # line of best fit... 
    #     bm, bc = np.polyfit(gens_range, bgp, 1)
    #     print(f"buy: m={bm}, c={bc}")

    #     # sell side
    #     sm, sc = np.polyfit(gens_range, sgp, 1)
    #     print(f"buy: m={sm}, c={sc}")

        
    #     # creating averages dictionary
    #     for gen in range(1, numgens+1):
    #         # buy side 
    #         if gen in bgenerational_bins:
    #             bgenerational_bins[gen] += bgp[gen-1]
    #         else:
    #             bgenerational_bins[gen] = bgp[gen-1]

    #         # sell side
    #         if gen in sgenerational_bins:
    #             sgenerational_bins[gen] += sgp[gen-1]
    #         else:
    #             sgenerational_bins[gen] = sgp[gen-1]

    #     _, (ax1, ax2) = plt.subplots(1,2)
    #     ax1.set_title("Buy Side")
    #     ax1.set(xlabel="Generation", ylabel="Profit")
    #     ax2.set_title("Sell Side")
    #     ax2.set(xlabel="Generation")

    #     ax1.plot(gens_range, bgp, '-g', label="Buyers")
    #     ax2.plot(gens_range, sgp, '-r', label="Seller")
    #     ax1.plot(gens_range, bother, '--g', label="Other Buyers")
    #     ax2.plot(gens_range, sother, '--r', label="Other Sellers")

    #     # ax1.plot(gens_range, gens_range*bm + bc, '--g', label="Buyers: Fitted")
    #     # ax2.plot(gens_range, gens_range*sm + sc, '--r', label="Sellers: Fitted")
    #     plt.show()


        

    
    # # line of best fit
    # # gradient and intersect

    # # # buy side
    # by = list(map(lambda x: x/10, bgenerational_bins.values()))
    # bm, bc = np.polyfit(gens_range, by, 1)
    # print(f"buy: m={bm}, c={bc}")

    # # sell side
    # sy = list(map(lambda x: x/10, sgenerational_bins.values()))
    # sm, sc = np.polyfit(gens_range, sy, 1)
    # print(f"buy: m={sm}, c={sc}")

    # _, (ax1, ax2) = plt.subplots(1,2)
    # ax1.plot(gens_range, gens_range*bm + bc, '--g')
    # ax2.plot(gens_range, gens_range*sm + sc, '--r')

    # # ax1.plot(gens_range, gens_range*bm + bc, '--g')
    # # ax2.plot(gens_range, gens_range*sm + sc, '--r')

    # # plotline(ax1, gens_range, bgp)







    # # ax.plot(gens_range, eq_prices, ":k", label="Equilibrium")
    # # plt.legend()
    # plt.show()

    # print('hello')