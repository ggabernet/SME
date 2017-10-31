import os
import sys
from collections import Counter

import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np
from Bio.SeqIO.FastaIO import FastaIterator
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})


def read_fasta(inputfile):
	"""Method for loading sequences from a FASTA formatted file and storing them into a list of sequences and names.

	:param inputfile: .fasta file with sequences and headers to read
	:return: lists of sequences and names.
	"""
	names = list()  # list for storing names
	sequences = list()  # list for storing sequences
	with open(inputfile) as handle:
		for record in FastaIterator(handle):  # use biopythons SeqIO module
			names.append(record.description)
			sequences.append(str(record.seq))
	return sequences, names


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


def _aa_count(pos):
	"""Method to calculate an aa count for each sequence position.

	:param seq: array containing amino acids for a given position.
	:return: dictionary containing the amino acid counts.
	"""
	count_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0,
				  'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
	count = Counter(pos)
	for aa, n in zip(count.keys(), count.values()):
		count_dict[aa] = n
	return count_dict


def _shannon_entropy(count):
	"""Method to calculate the shannon entropy given an amino acid count per position.

	:param count: (dict) dictionary containing the amino acid counts per position.
	:return: shannon entropy value scaled between 0 and 1.
	"""
	aa_probs = np.asarray(count.values()) / float(sum(count.values()))
	prob_log = aa_probs * np.log2(aa_probs)
	prob_log = np.nan_to_num(prob_log)
	return -sum(prob_log) / 4.32


class IterationAnalysis():
	def __init__(self, seqs, remove_parent=False):
		"""Class to analyse different properties of the offsprings of an Evolutionary Algorithm run.

		:param seqs: List of sequences, fasta file or csv file containing sequences for analysis.
		:param remove_parent: set to true if first sequence is the parent and it should be excluded from the entropy
		analysis.
		:Example:

		>>> from EA_tools import IterationAnalysis
		>>> run = IterationAnalysis('seqs.fasta', remove_parent=False)
		>>> run.calculate_entropy()
		>>> run.plot_alignment(colorscale='amphi')
		"""
		self.position_entropy = []
		self.sequences = []
		self.names = []
		self.parent_seq = []
		self.parent_name = []

		if type(seqs) == list:
			self.sequences = seqs
			self.names = []
		elif type(seqs) == np.ndarray:
			self.sequences = seqs.tolist()
			self.names = []
		elif os.path.isfile(seqs):
			if seqs.endswith('.fasta'):  # read .fasta file
				self.sequences, self.names = read_fasta(seqs)
			else:
				print("Sorry, currently only .fasta files can be read!")
		else:
			print("%s does not exist, is not a valid list of AA sequences or is not a valid sequence string" % seqs)

		if remove_parent:
			self.parent_seq = self.sequences.pop(0)
			self.parent_name = self.names.pop(0)

		char_mat = []
		for seq in self.sequences:
			schar = list(seq)
			char_mat.append(schar)
		self.char_mat = np.asarray(char_mat)

	def calculate_entropy(self):
		"""Method for calculating the shannon entropy for all the sequence positions.

		:return: shannon entropy scaled between 0 and 1 for all sequence positions in the attribute position_entropy.
		"""
		pos_entr = []
		for i in range(self.char_mat.shape[1]):
			count = _aa_count(self.char_mat[:, i])
			pos_entr.append(_shannon_entropy(count))
		self.position_entropy = np.asarray(pos_entr)
		return self.position_entropy

	def plot_alignment(self, colorscale='rainbow', filename="none"):
		""" Method to plot alignment of evolutionary algorithm offspring iteration together with its entropy values.

		:param colorscale: (str) color scale for coloring the amino acids. Available: 'rainbow', 'charge', 'polar',
		'simple', 'none', 'amphi',
		:param filename: (str) path where to store the figure. If None, plot is only shown.
		:return: plot of the alignment of the evolutionary algorithm iteration offspring and the position entropy.
		"""
		# calculate entropy for the plots
		self.calculate_entropy()

		# color mappings
		aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
		f_rainbow = np.asarray(
			['#3e3e28', '#ffcc33', '#b30047', '#b30047', '#ffcc33', '#3e3e28', '#80d4ff', '#ffcc33', '#0047b3',
			 '#ffcc33', '#ffcc33', '#b366ff', '#29a329', '#b366ff', '#0047b3', '#ff66cc', '#ff66cc', '#ffcc33',
			 '#ffcc33', '#ffcc33'])
		f_charge = np.asarray(
			['#000000', '#000000', '#ff4d94', '#ff4d94', '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff',
			 '#000000', '#000000', '#000000', '#000000', '#000000', '#80d4ff', '#000000', '#000000', '#000000',
			 '#000000', '#000000'])
		f_polar = np.asarray(
			['#000000', '#000000', '#80d4ff', '#80d4ff', '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff',
			 '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff', '#80d4ff', '#80d4ff', '#80d4ff', '#000000',
			 '#000000', '#000000'])
		f_simple = np.asarray(
			['#ffcc33', '#ffcc33', '#0047b3', '#0047b3', '#ffcc33', '#7f7f7f', '#0047b3', '#ffcc33', '#0047b3',
			 '#ffcc33', '#ffcc33', '#0047b3', '#ffcc33', '#0047b3', '#0047b3', '#0047b3', '#0047b3', '#ffcc33',
			 '#ffcc33', '#ffcc33'])
		f_none = np.asarray(['#ffffff'] * 20)
		f_amphi = np.asarray(
			['#ffcc33', '#29a329', '#b30047', '#b30047', '#f79318', '#80d4ff', '#0047b3', '#ffcc33', '#0047b3',
			 '#ffcc33', '#ffcc33', '#80d4ff', '#29a329', '#80d4ff', '#0047b3', '#80d4ff', '#80d4ff', '#ffcc33',
			 '#f79318', '#f79318'])
		t_rainbow = ['w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k']
		t_charge = ['w', 'w', 'k', 'k', 'w', 'w', 'k', 'w', 'k', 'w', 'w', 'w', 'w', 'w', 'k', 'w', 'w', 'w', 'w', 'w']
		t_polar = ['w', 'w', 'k', 'k', 'w', 'w', 'k', 'w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'k', 'k', 'w', 'w', 'w']
		t_simple = ['k', 'k', 'w', 'w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'k', 'k', 'w', 'w', 'w', 'w', 'k', 'k', 'k']
		t_none = ['k'] * 20
		t_amphi = ['k', 'k', 'w', 'w', 'w', 'k', 'w', 'k', 'w', 'k', 'k', 'k', 'w', 'k', 'w', 'k', 'k', 'k', 'w', 'w']

		if colorscale == 'rainbow':
			color = f_rainbow
			t_color = t_rainbow
		elif colorscale == 'charge':
			color = f_charge
			t_color = t_charge
		elif colorscale == 'polar':
			color = f_polar
			t_color = t_polar
		elif colorscale == 'simple':
			color = f_simple
			t_color = t_simple
		elif colorscale == 'none':
			color = f_none
			t_color = t_none
		elif colorscale == 'amphi':
			color = f_amphi
			t_color = t_amphi
		else:
			print 'Color scale not recognized'
			sys.exit()

		# Translating colours into numbers for plotting
		colour_dict = dict(zip(set(color), range(len(set(color)))))
		number = np.copy(color)
		for k, v in colour_dict.iteritems(): number[color == k] = v
		number = number.astype(int)

		aa_number = dict(zip(aa, number))
		t_dict = dict(zip(aa, t_color))

		# Translating amino acid matrix into numbers for plotting
		num_mat = np.copy(self.char_mat)

		for k, v in aa_number.iteritems():
			num_mat[self.char_mat == k] = v
		num_mat = num_mat.astype(int)

		plt.figure()
		gs = grd.GridSpec(2, 1, height_ratios=[8, 2])
		plt.subplots_adjust(left=0.12, bottom=0.08, right=0.85, top=0.92, wspace=0.01, hspace=0.08)

		# List of colors for plotting
		color = ListedColormap(set(color))

		# Plotting number matrix
		ax1 = plt.subplot(gs[0])
		ax1.matshow(num_mat, cmap=color, alpha=0.8, aspect="auto")

		# Annotating amino acids with correct color
		for i in xrange(num_mat.shape[1]):
			for j in xrange(num_mat.shape[0]):
				c = self.char_mat[j, i]
				ax1.text(i, j, c, va='center', ha='center', color=t_dict[c], size=12, fontweight='bold')

		ax1.set_yticks(np.arange(num_mat.shape[0]))
		ax1.set_xticks(range(4, num_mat.shape[1] + 1, 5))
		ax1.set_xticklabels(range(5, num_mat.shape[1] + 1, 5))
		ax1.set_yticklabels(self.names)
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.spines['left'].set_visible(False)
		ax1.tick_params(axis=u'both', which=u'both', length=0)

		# Subplot for entropy
		ax2 = plt.subplot(gs[1])

		x = np.arange(0, num_mat.shape[1], 1)
		ax2.bar(x, self.position_entropy, color='k', lw=0.4, align='center')

		ax2.spines['right'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax2.xaxis.set_ticks_position('bottom')
		ax2.yaxis.set_ticks_position('left')
		ax2.set_yticks([0, 1])
		ax2.set_xticklabels(range(5, num_mat.shape[1] + 1, 5))
		ax2.set_xticks(range(4, num_mat.shape[1] + 1, 5))
		plt.xlim(-0.5, (num_mat.shape[1] - 0.5))
		plt.ylim(0, 1)
		plt.xlabel("Position")
		plt.ylabel("Entropy")


		if filename == "none":
			plt.show()
		else:
			plt.savefig(filename)
