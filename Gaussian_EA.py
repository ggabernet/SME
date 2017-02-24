"""
Vespa Helix v2 (No Cys, Met) originally written by Gisbert Schneider in c++, 9 Jan 2016.
Rewritten in python by Gisela Gabernet, 24 Feb 2017.
"""
import sys
import numpy as np

print "\nVESPA Helix v2 (no Cys, Met) \n"

# TODO: implement argument checking.

# Setting random seed
np.random.seed(3)

def boxmuller():
    """
    Using the Box-Muller transformation it generates a gaussian distribution centered in 0 and with sigma = 1
    given two randomly generated numbers (i, j) in the interval [0, 1].
    :return: Normal distributed sample with expectation 0 and sigma 1.
    """
    i = np.random.random_sample(1) # picks random float [0, 1)
    j = np.random.random_sample(1) # picks random float [0, 1)
    return np.sqrt(-2.0 * np.log10(i)) * np.sin(2.0 * np.pi * j)

def mutate(parent, sigma):
    # Read grantham matrix
    mat = np.genfromtxt(matrixfile, delimiter='\t', dtype=float, skip_header=True)
    aa_order = np.genfromtxt(matrixfile, max_rows=1, dtype=str)
    print mat
    print mat.shape
    print aa_order

    #Scale each row
    row_max = np.max(mat, axis=1)
    row_min = np.min(mat, axis=1)
    mat_scaled = (mat - row_min) / (row_max - row_min)
    print mat_scaled
    print mat_scaled.shape

    for p in range(len(parent)):
        i = int(np.where(parent[p] == aa_order)[0]) # Translating the amino acid to a column in the grantham matrix.
        print p, i

        # Calculate the probability of mutation
        probs = np.exp(())

        # Making probability of mutation to M and C always 0.

# def main():
#     if len(sys.argv) != 5:
#         sys.exit("\nUSAGE: <seed> <lambda> <sigma> <matrixFile>\n\n")

# # Reading inputs from argv
# seed = str(sys.argv[1])
# lamb = int(sys.argv[2])  # Lambda
# sigma = float(sys.argv[3])
# matrixfile = str(sys.argv[4])

seed = "KLLKLLKKLLKLLK"
lamb = 10
sigma = 0.1
matrixfile = "grantham_matrix2.txt"


# TODO: check that matrix file can be opened, otherwise abort.

length = len(seed)
dist = 0
n = 0


print "\nSeed:  %s\nStrategy:   (1, %i)\nSigma:   %.2f\n\n(No)  (Dist)  (Sigma)  " \
      "(Sequence)       (Helix) (HelixContent)\n" %(seed, lamb, sigma)

print "%3.0f    %3.3f   %.2f    %s" %(n, dist, sigma, seed)

for n in range(1, lamb+1):
    box = boxmuller()
    sigmaOff = abs(sigma + box * sigma)
    print "boxmuller", box
    print "sigma", sigmaOff



    print "%3.0f    %3.3f   %.2f" %(n, dist, sigmaOff)

mutate(seed, sigma)


# if __name__ == '__main__':
#   main()