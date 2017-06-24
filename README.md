# Tensor Decomposition with Missing Indices

The implementation of the proposed method in our paper "Tensor Decomposition with Missing Indices", which will be published upcoming IJCAI 2017.

## Usage

You'll need to prepare the input tensor file (file format shown below), regularization parameter (lamb), and the number of feature dimensions (k).

```
$ python main.py -h
usage: main.py [-h] -i INFILE -o OUTFILE -l LAMB -k K [-n [NEPOCHS]] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        input tensor file
  -o OUTFILE, --outfile OUTFILE
                        output file
  -l LAMB, --lamb LAMB  weight of regularization
  -k K                  # feature dimensions
  -n [NEPOCHS], --nepochs [NEPOCHS]
                        # epochs (default=30)
  -v, --verbose         verbosity
```

## Input file

The input file must contain the data shape of the input tensor in the first line, and then the data follows.

```
$ cat data.dat
25297 2616 49 -1
0 0 1 1
12 1 3 1
8 -1 14 1
2 9 -1 1
5 1 5 1
```

If the mode of the input tensor is three, then each line contains four numbers separated by space.

In the first line, the first three numbers define the number of dimensions of each mode, and the fourth number will be ignored, which is needed for implementation reason :(

After that, in each line, the first three numbers show the index for each mode, and the last (fourth) number shows the value of the tensor in the corresponding position.

If the index is -1, then that index is ***missing***, in other words, the position of the corresponding value is not exactly determined.

## Reference
Yuto Yamaguchi, Kohei Hayashi, "Tensor Decomposition with Missing Indices", International Joint Conference on Artificial Intelligence (IJCAI), Melbourne, Australia, Aug 19-25, 2017.
