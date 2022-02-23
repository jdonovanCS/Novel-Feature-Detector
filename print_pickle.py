import pickle

with open("solutions_over_time.pickle", 'rb') as f:
    pickled = pickle.load(f)
    for k,v in pickled.items():
        print(len(v[0][0][2][0][0][0]))
    # print(pickled)