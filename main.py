import pickle
import numpy as np
import argparse
from lib.model import MODEL

def read_data(filepath):
    return np.genfromtxt(filepath, dtype=float)

def dump_model(outfile, model):
    with open(outfile, "wb") as f:
        pickle.dump(model, f)

p = argparse.ArgumentParser()
p.add_argument("-i", "--infile", help="input tensor file", type=str, required=True)
p.add_argument("-o", "--outfile", help="output file", type=str, required=True)
p.add_argument("-l", "--lamb", help="weight of regularization", type=float, required=True)
p.add_argument("-k", help="# feature dimensions", type=int, required=True)
p.add_argument("-n", "--nepochs", help="# epochs (default=30)", type=int, nargs='?', default=30)
p.add_argument("-v", "--verbose", help="verbosity", action='store_true')
args = p.parse_args()

data = read_data(args.infile)
data_shape = data[0,:-1].astype(int)  # header
X = data[1:,:]

model = MODEL(args.k, args.lamb, data_shape, args.nepochs, args.verbose)
model.fit(X)
dump_model(args.outfile, model)
