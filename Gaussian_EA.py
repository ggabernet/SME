"""
Vespa v2 (No Cys, Met) originally written by Gisbert Schneider in c++, 9 Jan 2016.
Rewritten in python by Gisela Gabernet including , 27 Feb 2017
"""
import sys
import numpy as np



# TODO: implement argument checking.

# Setting random seed
# TODO: handle seed properly
np.random.seed(3)

def main():
    if len(sys.argv) != 5:
        sys.exit("\nUSAGE: <seed> <lambda> <sigma> <matrixFile>\n\n")

    print "\nVESPA Helix v2 (no Cys, Met) \n\n Calculating... \n\n"

    # Reading inputs from argv
    seed = str(sys.argv[1])
    lamb = int(sys.argv[2])  # Lambda
    sigma = float(sys.argv[3])
    matrixfile = str(sys.argv[4])

    # seed = "KLLKLLKKLLKLLK"
    # lamb = 10
    # sigma = 0.05
    # matrixfile = "grantham_matrix2.txt"


    # TODO: check that matrix file can be opened, otherwise abort.

    length = len(seed)
    dist = 0
    n = 0

    with open("vespa_run.txt", mode='w') as f:
        f.write("\nVESPA Helix v2 (no Cys, Met) \n"
                "\nSeed:  %s\nStrategy:   (1, %i)\nSigma:   %.2f\n\n"
                "(No)  (Dist)  (Sigma)  (Sequence)\n" %(seed, lamb, sigma))

        f.write("%3.0f    %3.3f   %.2f    %s\n" %(n, dist, sigma, seed))

        for n in range(1, lamb+1):
            box = boxmuller()
            sigmaOff = abs(sigma + box * sigma)

            offspring, dist = mutate(seed, sigma, matrixfile)

            f.write("%3.0f    %3.3f   %.2f    %s\n" %(n, dist, sigmaOff, offspring))

    print "Run completed, results are stored in vespa_run.txt"


def boxmuller():
    """
    Using the Box-Muller transformation it generates a gaussian distribution centered in 0 and with sigma = 1
    given two randomly generated numbers (i, j) in the interval [0, 1].
    :return: Normal distributed sample with expectation 0 and sigma 1.
    """
    i = np.random.random_sample(1) # picks random float [0, 1)
    j = np.random.random_sample(1) # picks random float [0, 1)
    return np.sqrt(-2.0 * np.log10(i)) * np.sin(2.0 * np.pi * j)


def mutate(parent, sigma, matrixfile):
    """
    Mutates *parent* sequence according to probabilities derived from the Grantham matrix, which have been decayed using
    the boltzmann function in the following way.

    .. math::

        P(i -> j) = exp(- dfrac{d_{ij}^2}{2 \sigma^2}) / sum_j exp(- dfrac{d_{ij}^2}{2 \sigma^2})

    where :math:`d_{ij}`is the pair-wise amino acid distance in the provided matrix, scaled to a common interval to
    obtain pseudo-probabilities for each residue position in a peptide, and :math:\sigma` is the provided sigma.

    For more information refer to: J. A. Hiss & G. Schneider. Peptide Design by Nature-Inspired Algorithms, in
    De novo Molecular Design. pp 444 - 448. 2014 Wiley-VCH.

    :param parent: {str} parent sequence.
    :param sigma: {float} sigma value for the width of the gaussian distribution for the distance to the parent residue.
    :param matrixfile: {str} path to text file where distance matrix is stored.
    :return: offspring sequence as string, euclidean distance to parent as float.
    """
    # Read grantham matrix
    mat = np.genfromtxt(matrixfile, delimiter='\t', dtype=float, skip_header=True)
    aa_order = np.genfromtxt(matrixfile, max_rows=1, dtype=str)

    # Scale each row
    row_max = np.max(mat, axis=1)
    row_min = np.min(mat, axis=1)
    mat_scaled = (mat - row_min) / (row_max - row_min)
    offspring = []
    dist_aa = 0

    for p in range(len(parent)):
        i = int(np.where(parent[p] == aa_order)[0]) # Translating the amino acid to a column in the grantham matrix.

        # Calculate the probability of mutation
        probs = np.exp((-(mat_scaled[i, :] * mat_scaled[i, :])/(2 * sigma * sigma)) /
                       np.sum(np.exp(-(mat_scaled[i, :] * mat_scaled[i, :])/(2 * sigma * sigma))))

        # Making probability of mutation to M and C always 0.
        mc0 = np.array([1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.])
        probs = probs * mc0 / (np.sum(probs * mc0))

        # Picking amino acid:
        aa_mut = np.random.choice(aa_order, 1, p=probs)
        aa_mut_idx = int(np.where(aa_mut == aa_order)[0])
        offspring.extend(aa_mut)
        dist_aa += (mat_scaled[i, aa_mut_idx]) ** 2

    return ''.join(offspring), np.sqrt(dist_aa)



if __name__ == '__main__':
  main()