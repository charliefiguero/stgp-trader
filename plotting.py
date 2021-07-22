import os
import glob
import csv
from datetime import datetime
import pickle


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

def plot_tran_price():
    with open('Test00tapes.csv', 'r') as infile:
        reader = csv.reader(infile)
        timeprice = [(row[2], row[3]) for row in reader]
        times = [item[0] for item in timeprice]
        prices = [item[1] for item in timeprice]

    print(len(times))
    print(len(prices))
    output = sns.lineplot(x=times, y=prices)
    output.get_figure().savefig(f'timeprices{datetime.now()}.png')
    # plt.show()


if __name__ == "__main__":
    # plot_stats()
    # plot_tran_price()
    plot_hof()



    
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