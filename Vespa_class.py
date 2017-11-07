"""
ETH Zurich | Gisela Gabernet

Simulated molecular evolution algorithm (SME)


"""

import sys
import numpy as np
import os
import time


def main():
    if len(sys.argv) < 7 or len(sys.argv) > 8:
        sys.exit("\nUSAGE: <seed> <lambda> <sigma> <sigma strategy (G/T)> "
                 "<matrixFile> <Non-desired aa> "
                 "<Random_seed (opt)>\n\n"
                 "Example: python Vespa.py KLLKLLKKLLKLLK 10 0.05 G grantham_matrix.txt CM 3\n\n"
                 "Seed Klak14, lambda 10, sigma 0.05, offspring sigma gaussian distributed around parent sigma, "
                 "use grantham_matrix.txt for mutation probabilities, exclude Cys and Met from offspring (type 0 if "
                 "no aa is to be excluded), "
                 "set random_seed = 3 (optional argument, makes results reproducible by setting a random seed).\n\n"
                 "Offspring sigma modalities available:\n"
                 "- 'G' (Gaussian distributed): offspring sigma gaussian distributed with SD seed sigma and "
                 "centered on seed sigma\n"
                 "- 'T' (one-third strategy): offspring sigma with probability 1/3 0.7*sigma, "
                 "with probability 1/3 sigma, with probability 1/3 1.3 * sigma")

    print "\nVESPA v3 \n\n Calculating... \n"

    # Reading inputs from argv.
    seed = str(sys.argv[1])
    lamb = int(sys.argv[2])  # Lambda
    sigma = float(sys.argv[3]) # Sigma
    sigma_strategy = str(sys.argv[4])
    matrixfile = str(sys.argv[5])
    no_aa = list(sys.argv[6])

    # Setting random seed if provided.
    if len(sys.argv) == 8:
        random_seed = int(sys.argv[7])
        np.random.seed(random_seed)

    # Checking that the matrix file exists.
    if not os.path.exists(matrixfile):
        sys.exit("\n Matrix file does not exist under the provided path.")


    with open(time.strftime("%Y-%m-%d-%H%M-vespa_run.txt"), mode='w') as f:
        f.write("\nVESPA v3\n"
                "\nSeed:  %s\nStrategy:   (1, %i)\nSigma:   %.2f\nOffspring sigma strategy: %s"
                "\nUndesired aa:    %s\n" % (seed, lamb, sigma, sigma_strategy, ''.join(no_aa)))
        if len(sys.argv) == 8:
            f.write("Random Seed:   %i\n" % (random_seed))
        f.write("\n(No)  (Dist)  (Sigma)  (Sequence)\n" )

        f.write("%3.0f    %3.3f   %.2f    %s\n" % (0, 0, sigma, seed))

        for n in range(1, lamb+1):

            # Choosing offspring sigma.
            if sigma_strategy == "G": # gaussian strategy with mean parent sigma and SD parent sigma.
                sigma_off = abs(sigma + boxmuller() * sigma)

            elif sigma_strategy == "T": # one-third strategy
                sigma_off = np.random.choice([0.7 * sigma, sigma, 1.3 * sigma])

            else:
                sys.exit("Unknown Offspring sigma strategy, please choose from:"
                         "- G (Gaussian distributed): offspring sigma gaussian distributed with SD seed sigma and "
                         "centered on seed sigma\n"
                         "- T (one-third strategy): offspring sigma with probability 1/3 0.7*sigma, "
                         "with probabiliyt 1/3 sigma, with probability 1/3 1.3 * sigma")

            # Generating mutant offspring and calculating the euclidean distance to the parent.
            offspring, dist = mutate(seed, sigma, matrixfile, no_aa)

            f.write("%3.0f    %3.3f   %.2f    %s\n" % (n, dist, sigma_off, offspring))

    print time.strftime("Run completed, results are stored in %Y-%m-%d-%H%M-vespa_run.txt")


def _boxmuller():
    """
    The Box-Muller transformation picks a number from a gaussian distribution centered in 0 and with sigma = 1
    given two randomly generated numbers (i, j) in the interval [0, 1].
    :return: (float) number picked from normal distribution with expectation 0 and sigma 1.
    """
    i = np.random.random_sample(1)  # picks random float [0, 1)
    j = np.random.random_sample(1)  # picks random float [0, 1)
    return np.sqrt(-2.0 * np.log10(i)) * np.sin(2.0 * np.pi * j)

class SME():

    def __init__(self, parent, offspring_number, n_generations, sigma, sigma_strategy, fitness_function, matrix_file, aa_excluded):
        """
        Initialization of a new generation for a simulated molecular evolution iteration.
        :param parent: {str} peptide sequence of the parent peptide.
        :param offspring_number: {int} number of offsprings for the generation.
        :param generations: {int} number of generations.
        :param sigma: {float} sigma value for the probability of mutation distance.
        :param sigma_strategy: {str: G/T} sigma strategy for the sigma evolution across generations.
        :param fitness_function: {} fitness function to use as  offspring selection.
        :param aa_matrix: {path to matrix file} amino acid similarity matrix, when 'grantham' is chosen, .
        :param aa_excluded: amino acids to exclude in generating the mutations.
        """
        self.first_parent = parent
        self.parent = parent
        self.off_num = offspring_number
        self.sigma_start = sigma
        self.sigma = sigma
        self.sigma_strategy = sigma_strategy
        self.n_generations = n_generations
        self.aa_excluded = aa_excluded
        self.offsprings = []
        self.fitness_function = []

        if not os.path.exists(matrix_file):
            sys.exit("\n Matrix file does not exist under provided path.")

        aa_matrix = np.genfromtxt(matrix_file, delimiter='\t', dtype=float, skip_header=True)
        self.aa_matrix = (aa_matrix - np.min(aa_matrix, axis=1) /
                          (np.max(aa_matrix, axis=1) - np.min(aa_matrix, axis=1))

        self.aa_order = np.genfromtxt(matrix_file, delimiter='\t', max_rows=1, dtype=str)

    def _mutate(self):
        """
        Mutates *parent* sequence according to probabilities derived from the provided matrixfile, which have been scaled
        row-wise and using the boltzmann function with decay parameter sigma in the following way:

        .. math::

            P(i -> j) = exp(- dfrac{d_{ij}^2}{2 \sigma^2}) / sum_j exp(- dfrac{d_{ij}^2}{2 \sigma^2})

        where :math:`d_{ij}`is the pair-wise amino acid distance in the provided matrix, scaled to a common interval to
        obtain pseudo-probabilities for each residue position in a peptide, and :math:\sigma` is the provided sigma.

        For more information refer to: J. A. Hiss & G. Schneider. Peptide Design by Nature-Inspired Algorithms, in
        De novo Molecular Design. pp 444 - 448. 2014 Wiley-VCH.

        :param parent: {str} parent sequence.
        :param sigma: {float} sigma value for the width of the gaussian distribution for the distance to the parent residue.
        :param matrixfile: {str} path to text file where distance matrix is stored. Data needs to be tab separated.
        First row contains amino-acid headers in one-letter code.
        :param no_aa: {list} list of aa in 1-letter code that are chosen to be excluded from offspring.
        :return: offspring sequence as string, euclidean distance to parent as float.
        """

        mat_scaled = self.aa_matrix
        aa_order = self.aa_order
        offspring = []
        dist_aa = 0

        for p in range(len(self.parent)):
            i = int(
                np.where(self.parent[p] == aa_order)[
                    0])  # Translating the amino acid to a column index in the grantham matrix.

            # Calculate the probability of mutation
            probs = np.exp((-(mat_scaled[i, :] * mat_scaled[i, :]) / (2 * self.sigma * self.sigma)) /
                           np.sum(np.exp(-(mat_scaled[i, :] * mat_scaled[i, :]) / (2 * self.sigma * self.sigma))))

            # Making probability of mutation to undesired aa in "no_aa" always 0.
            mc0 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
            if len(self.aa_excluded) == 1 and self.aa_excluded[0] == '0':
                pass
            elif len(self.aa_excluded) >= 1:
                for aa in self.aa_excluded:
                    idx_aa = int(np.where(aa == aa_order)[0])
                    mc0[idx_aa] = 0
            else:
                sys.exit("Non-desired aa argument invalid. Please type aa in 1-letter code without separation that"
                         "you wish to be excluded in the mutations, or type 0 if you don't wish to exclude any aa.")
            probs = probs * mc0 / (np.sum(probs * mc0))

            # Picking amino acid:
            aa_mut = np.random.choice(aa_order, 1, p=probs)
            aa_mut_idx = int(np.where(aa_mut == aa_order)[0])
            offspring.extend(aa_mut)
            dist_aa += (mat_scaled[i, aa_mut_idx]) ** 2

            # Choosing offspring sigma.
            if self.sigma_strategy == "G":  # gaussian strategy with mean parent sigma and SD parent sigma.
                sigma_off = abs(self.sigma + _boxmuller() * self.sigma)

            elif self.sigma_strategy == "T":  # one-third strategy
                sigma_off = np.random.choice([0.7 * self.sigma, self.sigma, 1.3 * self.sigma])

            else:
                sys.exit("Unknown Offspring sigma strategy, please choose from:"
                         "- G (Gaussian distributed): offspring sigma gaussian distributed with SD seed sigma and "
                         "centered on seed sigma\n"
                         "- T (one-third strategy): offspring sigma with probability 1/3 0.7*sigma, "
                         "with probabiliyt 1/3 sigma, with probability 1/3 1.3 * sigma")

        return ''.join(offspring), np.sqrt(dist_aa), sigma_off

    def iteration(self):

        sigma = self.sigma

        # Mutating to new offspring, calculating distance and sigma.
        off_generation = []
        for i in range(self.off_num):
            off, dist, sigma = self._mutate()
            off_generation.append((off,dist,sigma))
        self.offsprings.append(off_generation)





if __name__ == '__main__':
  main()
