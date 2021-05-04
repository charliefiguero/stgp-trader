import jsonpickle
import pandas as pd


def read_pickle(fname):
    with open(fname, 'r') as file:
        data = file.read()
        return jsonpickle.decode(data)



if __name__ == "__main__":
    experiment_data = read_pickle("stgp_csvs/improvements/2021-05-05 00:00:36.191503")
    print(experiment_data['traders_data'])

    exp_df = pd.DataFrame.from_dict(experiment_data)
    print(exp_df)

