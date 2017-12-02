"""
ETH Zurich | Gisela Gabernet

Simulated molecular evolution algorithm (SME)


"""

import sys
import numpy as np
import os


def save_fasta(filename, sequences, names=None):
	"""Method for saving sequences (and names) into a fasta file.

	:param filename: {str} output filename (ending .fasta)
	:param sequences: {list} sequences to be saved to file
	:param names: {list} whether sequence names from self.names should be saved as sequence identifiers
	:return: a FASTA formatted file containing the generated sequences
	"""
	if os.path.exists(filename):
		os.remove(filename)  # remove outputfile, it it exists

	with open(filename, 'w') as o:
		for n, seq in enumerate(sequences):
			if names:
				o.write('>' + str(names[n]) + '\n')
			else:
				o.write('>Seq_' + str(n) + '\n')
			o.write(seq + '\n')


def _boxmuller():
	"""The Box-Muller transformation picks a number from a gaussian distribution centered in 0 and with sigma = 1
	given two randomly generated numbers (i, j) in the interval [0, 1].

	:return: {float} number picked from normal distribution with expectation 0 and sigma 1.
	"""
	i = np.random.random_sample(1)  # picks random float [0, 1)
	j = np.random.random_sample(1)  # picks random float [0, 1)
	return np.sqrt(-2.0 * np.log10(i)) * np.sin(2.0 * np.pi * j)


def mutate(seq, sigma, sigma_strategy, aa_matrix, aa_order, aa_excluded):
	"""Mutates *parent* sequence according to probabilities derived from the provided matrixfile, which have been scaled
	row-wise and using the boltzmann function with decay parameter sigma in the following way:

	.. math::

		P(i -> j) = exp(- dfrac{d_{ij}^2}{2 \sigma^2}) / sum_j exp(- dfrac{d_{ij}^2}{2 \sigma^2})

	where :math:`d_{ij}`is the pair-wise amino acid distance in the provided matrix, scaled to a common interval to
	obtain pseudo-probabilities for each residue position in a peptide, and :math:\sigma` is the provided sigma.

	For more information refer to: J. A. Hiss & G. Schneider. Peptide Design by Nature-Inspired Algorithms, in
	De novo Molecular Design. pp 444 - 448. 2014 Wiley-VCH.


	:param seq: {str} sequence to mutate.
	:param sigma: {float} sigma value for the mutation probability distribution.
	:param aa_matrix: {np array} normalized aa similarity matrix.
	:param aa_order: {np array} order of the amino acids in the aa similarity matrix.
	:param aa_excluded: {list} list of the amino acids that are desired to be excluded from the offspring.
	:return:
	"""

	mat = aa_matrix
	aa_order = aa_order
	offspring = []
	dist_aa = 0

	for p in range(len(seq)):
		i = int(np.where(seq[p] == aa_order)[0])  # Translating the amino acid to a column index in the grantham matrix.

		# Calculate the probability of mutation
		probs = np.exp((-(mat[i, :] * mat[i, :]) / (2 * sigma * sigma)) /
					   np.sum(np.exp(-(mat[i, :] * mat[i, :]) / (2 * sigma * sigma))))

		# Making probability of mutation to undesired aa in "no_aa" always 0.
		mc0 = np.ones(20)
		if len(aa_excluded)==1 and aa_excluded == '0':
			pass
		elif len(aa_excluded) >= 1:
			for aa in aa_excluded:
				idx_aa = int(np.where(aa == aa_order)[0])
				mc0[idx_aa] = 0
		else:
			sys.exit("Non-desired aa argument invalid. Please type aa in 1-letter code without separation that "
					 "you wish to be excluded in the mutations, or type 0 if you don't wish to exclude any aa.")
		probs = probs * mc0 / (np.sum(probs * mc0))

		# Picking amino acid:
		aa_mut = np.random.choice(aa_order, 1, p=probs)
		aa_mut_idx = int(np.where(aa_mut == aa_order)[0])
		offspring.extend(aa_mut)
		dist_aa += (mat[i, aa_mut_idx]) ** 2

		# Choosing offspring sigma.
		if sigma_strategy == "Gaussian":  # gaussian strategy with mean parent sigma and SD parent sigma.
			sigma_off = abs(sigma + _boxmuller() * sigma)

		elif sigma_strategy == "Thirds":  # one-third strategy
			sigma_off = np.random.choice([0.7 * sigma, sigma, 1.3 * sigma])

		elif sigma_strategy == "Equal":  # sigma remains constant
			sigma_off = sigma

		else:
			sys.exit("Unknown Offspring sigma strategy, please choose from:\n"
					 "- Equal (default): sigma remains constant\n"
					 "- Gaussian (Gaussian distributed): offspring sigma gaussian distributed with SD seed sigma and "
					 "centered on seed sigma\n"
					 "- Thirds (one-third strategy): offspring sigma with probability 1/3 0.7*sigma, "
					 "with probability 1/3 sigma, with probability 1/3 1.3 * sigma\n")

	return ''.join(offspring), np.sqrt(dist_aa), sigma_off


class SME(object):
	def __init__(self, parent, n_offspring, sigma, matrix_file,
				 sigma_strategy="Equal", aa_excluded=None, random_seed=None):
		"""Initialization of a new generation for a simulated molecular evolution iteration.

		:param parent: {str} peptide sequence of the parent peptide.
		:param offspring_number: {int} number of offspring for the generation.
		:param n_generations: {int} number of generations.
		:param sigma: {float} sigma value for the probability of mutation distance.
		:param sigma_strategy: {str: Gaussian/Thirds} sigma strategy for the sigma evolution across generations.
		:param fitness_function: {} fitness function to use as offspring selection.
		:param matrix_file: {path to matrix file} amino acid similarity matrix, the first row needs to be
		the amino acid order of the columns in the matrix.
		:param aa_excluded: {list or string} amino acids to exclude in generating the mutations, in one letter code,
		as string e.g. "CM" or list of strings e.g. ["C","M"].
		"""
		self.parent_first = parent
		self.parent = parent
		self.n_offspring = n_offspring
		self.sigma_start = sigma
		self.sigma = sigma
		self.sigma_strategy = sigma_strategy
		self.n_iterations = 0
		if aa_excluded:
			self.aa_excluded = aa_excluded
		else:
			self.aa_excluded = '0'

		self.offspring = []
		self.offspring_seq = []

		if random_seed:
			self.random_seed = random_seed
			np.random.seed(random_seed)
		else:
			self.random_seed = None

		if not os.path.exists(matrix_file):
			sys.exit("\n Matrix file does not exist under provided path.")
		aa_matrix = np.genfromtxt(matrix_file, delimiter='\t', dtype=float, skip_header=True)
		self.aa_matrix = (aa_matrix - np.min(aa_matrix, axis=1)) / \
						 (np.max(aa_matrix, axis=1) - np.min(aa_matrix, axis=1))
		self.aa_order = np.genfromtxt(matrix_file, delimiter='\t', max_rows=1, dtype=str)

	def variation(self):
		"""Method for producing a new iteration of simulated molecular evolution.

		:return: Appended new offspring sequences, calculated distances and offspring sigma
		for the iteration in attribute offspring.
		"""
		self.n_iterations += 1

		# Mutating to new offspring, calculating distance and sigma.
		off_generation = [(0, self.parent, 0, self.sigma)]
		for i in range(1, self.n_offspring + 1):
			off, dist, sigma = mutate(self.parent, self.sigma, self.sigma_strategy, self.aa_matrix, self.aa_order,
									  self.aa_excluded)
			off_generation.append((i, off, dist, sigma))
			self.offspring_seq.append(off)
		self.offspring.append(off_generation)

	def update_parent(self, parent):
		"""Method to manually select a new parent for the following round of iteration.

		:param parent: {str} sequence of the parent peptide.
		:return: updated parent sequence in self.parent
		"""

		self.parent = parent

		if self.parent not in self.offspring_seq:
			print "\nWARNING: the selected parent sequence is not registered in the offspring\n"

	def selection(self, fitness_function):
		"""Method for selecting the best offspring according to a fitness function.

		--> need to update parent and sigma in self.parent and self.sigma.

		:param fitness_function: fitness function.
		:return: updated parent and sigma in attibutes self.parent and self.sigma.
		"""

	def iterations(self, n, fitness_function):
		"""Method to produce several SME iterations according to the provided fitness function.

		--> need to update parent and sigma in self.parent and self.sigma.

		:param n: number of iterations.
		:param fitness_function: fitness function.
		:return: Appended new offspring sequences, calculated distances and offspring sigma
		for the iteration in attribute offspring.
		"""

	def save_offspring(self, filename):
		"""Method to save the offspring sequences in a text file and fasta file.

		:return: text file and fasta file with the offspring from all the iterations.
		"""

		with open(filename + ".txt", mode='w') as f:
			f.write("\nVESPA v3\n"
					"\nSeed:  %s\nStrategy:   (1, %i)\nSigma:   %.2f\nOffspring sigma strategy: %s"
					"\nUndesired aa:    %s\n" % (self.parent_first, self.n_offspring, self.sigma_start,
												 self.sigma_strategy, ''.join(self.aa_excluded)))
			if self.random_seed:
				f.write("Random Seed:   %i\n" % (self.random_seed))

			names = []
			seqs = []

			f.write("\nName\tDist\tSigma\tSequence\n")
			for i, iter in enumerate(self.offspring):
				for o, off in enumerate(iter):
					name = ''.join([str(i + 1), '.', str(o)])
					names.append(name)
					seqs.append(off[1])
					f.write("%s\t%3.3f\t%.2f\t%s\n" % (name, off[2], off[3], off[1]))

			save_fasta(filename + '.fasta', seqs, names)
