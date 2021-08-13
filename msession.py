import csv
import matplotlib.pyplot as plt

def genprofit(duration: int, num_gens: int, filename):

    with open(filename, 'r') as infile:
        reader = list(csv.reader(infile))

    time_per_gen = duration / num_gens
    eq_price = 100

    trader_profit = {}

    # 1 trader per row
    for row in reader:
        tid = row[1][1:]
        ttype = row[2]
        num_trades = row[3]
        trades = [list(map(float, x[1:].split(' '))) for x in row[4:]]

        trader_profit[tid] = {}

        gen = 0
        for t in trades:
            time = t[0]
            profit = t[1]

            while time > gen * time_per_gen: 
                gen += 1
                trader_profit[tid][gen] = []
            
            trader_profit[tid][gen].append(profit)

        # create generation arrays for generations with no trades
        if len(trader_profit[tid]) < num_gens:
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


def plotline(ax, x, y):
    ax.plot(x, y, "-g", label="GP Buyers")
    ax.plot(x, y)

if __name__ == "__main__":

    duration = 20000
    numgens = 40
    eqprice = 100

    _, (ax1, ax2) = plt.subplots(1,2)

    ax1.set_title("Buy Side")
    ax1.set(xlabel="Generation", ylabel="Price")


    ax2.set_title("Sell side")
    ax2.set(xlabel="Generation", ylabel="Price")

    gens_range = range(1, numgens+1)
    eq_prices = [eqprice] * len(gens_range)
    
    for i in range(10):
        path = "standard_csvs/"
        fname = path+f"Test{i:02}profit.csv"

        bgp, sgp, bother, sother = genprofit(duration, numgens, fname)
        plotline(ax1, gens_range, bgp)
        plotline(ax2, gens_range, sgp)









    # ax.plot(gens_range, eq_prices, ":k", label="Equilibrium")
    # plt.legend()
    plt.show()

    print('hello')


    