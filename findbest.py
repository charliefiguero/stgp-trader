""" Finds the expression with maximum profit. """

import csv

if __name__ == '__main__':
    boffspringf = "experiments/shvr_vs_shvr50/boffspring0.txt"
    soffspringf = "experiments/shvr_vs_shvr50/soffspring0.txt"

    with open(boffspringf, 'r') as infile:
        blines = infile.readlines()
    with open(soffspringf, 'r') as infile:
        slines = infile.readlines()

    max = 0
    tree = None
    generation = None

    current_gen = 0
    # boffspring
    for l in blines:
        if "Gen" in l:
            current_gen += 1

        parts = l.split(":")
        if parts[0] == '\n':
            continue

        try:
            profit = float(parts[0])
            expr = parts[1][1:]
            
            if profit > max:
                max = profit
                tree = expr
                generation = current_gen
        except:
            print(parts)
    # soffspring
    current_gen = 0
    for l in slines:
        if "Gen" in l:
            current_gen += 1

        parts = l.split(":")
        if parts[0] == '\n':
            continue

        try:
            profit = float(parts[0])
            expr = parts[1][1:]
            
            if profit > max:
                max = profit
                tree = expr
                generation = current_gen
        except:
            print(parts)



    print(f"profit:", max)
    print(f"generation:", generation)
    print(f"expression", tree)