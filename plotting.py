import os
import glob

import jsonpickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


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

if __name__ == "__main__":
    list_of_files = glob.glob('stgp_csvs/improvements/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f'reading file: ', latest_file, '\n')
    
    experiment_data = read_pickle(latest_file)
    exp_df = pd.DataFrame.from_dict(experiment_data)


    # calculate data
    gen_profits = gen_profits(exp_df['traders_data'])

    # plotting
    print(gen_profits)
    plot_gen_profits(gen_profits)

    # print(traders_gen_profits(exp_df['traders_data'][0]))
    # print(gen_profits(exp_df['traders_data']))