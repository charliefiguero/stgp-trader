import csv
import matplotlib.pyplot as plt
import numpy as np

def genprofit(duration: int, num_gens: int, filename):

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

        trader_profit[tid] = {'ttype':ttype}
        # trader_profit[tid] = {-1:ttype}

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
    BOTHER_keys = [x for x in trader_profit.keys() if "B" in x and x not in BSTGP_keys and trader_profit[x]['ttype'] == "ZIP"]
    SOTHER_keys = [x for x in trader_profit.keys() if "S" in x and x not in SSTGP_keys and x not in BSTGP_keys and trader_profit[x]['ttype'] == "ZIP"]



    def group_gen_profit(keys, gen):
        total_profit = 0
        for tkey in keys:
            print(tkey)
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
    ax.plot(x, y)

if __name__ == "__main__":

    duration = 20000
    numgens = 40
    eqprice = 100

    # _, (ax1, ax2) = plt.subplots(1,2)

    # ax1.set_title("Buy Side")
    # ax1.set(xlabel="Generation", ylabel="Profit")


    # ax2.set_title("Sell Side")
    # ax2.set(xlabel="Generation")

    gens_range = range(1, numgens+1)
    eq_prices = [eqprice] * len(gens_range)
    
    # buy side
    bgenerational_bins = {}
    # sell side
    sgenerational_bins = {}

    # _, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(3,3)

    x=0
    y=0
    for i in range(10):
        print(f"Session: {i}")


        path = "experiments/competitive/"
        fname = path+f"Test{i:02}profit.csv"

        bgp, sgp, bother, sother = genprofit(duration, numgens, fname)



        # line of best fit... 
        bm, bc = np.polyfit(gens_range, bgp, 1)
        print(f"buy: m={bm}, c={bc}")

        # sell side
        sm, sc = np.polyfit(gens_range, sgp, 1)
        print(f"buy: m={sm}, c={sc}")

        
        # creating averages dictionary
        for gen in range(1, numgens+1):
            # buy side 
            if gen in bgenerational_bins:
                bgenerational_bins[gen] += bgp[gen-1]
            else:
                bgenerational_bins[gen] = bgp[gen-1]

            # sell side
            if gen in sgenerational_bins:
                sgenerational_bins[gen] += sgp[gen-1]
            else:
                sgenerational_bins[gen] = sgp[gen-1]

        _, (ax1, ax2) = plt.subplots(1,2)
        ax1.set_title("Buy Side")
        ax1.set(xlabel="Generation", ylabel="Profit")
        ax2.set_title("Sell Side")
        ax2.set(xlabel="Generation")

        ax1.plot(gens_range, bgp, '-g', label="Buyers")
        ax2.plot(gens_range, sgp, '-r', label="Seller")
        ax1.plot(gens_range, bother, '--g', label="Other Buyers")
        ax2.plot(gens_range, sother, '--r', label="Other Sellers")

        # ax1.plot(gens_range, gens_range*bm + bc, '--g', label="Buyers: Fitted")
        # ax2.plot(gens_range, gens_range*sm + sc, '--r', label="Sellers: Fitted")
        plt.show()


        

    
    # line of best fit
    # gradient and intersect

    # # buy side
    by = list(map(lambda x: x/10, bgenerational_bins.values()))
    bm, bc = np.polyfit(gens_range, by, 1)
    print(f"buy: m={bm}, c={bc}")

    # sell side
    sy = list(map(lambda x: x/10, sgenerational_bins.values()))
    sm, sc = np.polyfit(gens_range, sy, 1)
    print(f"buy: m={sm}, c={sc}")

    _, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(gens_range, gens_range*bm + bc, '--g')
    ax2.plot(gens_range, gens_range*sm + sc, '--r')

    # ax1.plot(gens_range, gens_range*bm + bc, '--g')
    # ax2.plot(gens_range, gens_range*sm + sc, '--r')

    # plotline(ax1, gens_range, bgp)







    # ax.plot(gens_range, eq_prices, ":k", label="Equilibrium")
    # plt.legend()
    plt.show()

    print('hello')


    