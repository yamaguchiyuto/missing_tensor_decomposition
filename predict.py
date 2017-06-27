import pickle
import argparse
from lib.model import MODEL

def load_model(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

p = argparse.ArgumentParser()
p.add_argument("-m", "--modelfile", help="input model file path", type=str, required=True)
p.add_argument("-t", "--testfile", help="input test data file path", type=str, required=True)
args = p.parse_args()

model = load_model(args.modelfile)

for line in open(args.testfile, 'r'):
    index = list(map(int, line.split(' ')))
    print(model.predict(index))
