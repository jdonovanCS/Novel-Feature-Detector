import pickle

with open("solutions_over_time.pickle", 'rb') as f:
    pickled = pickle.load(f)
    print(pickled)