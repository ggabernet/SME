import numpy as np
from Bio.SeqIO.FastaIO import FastaIterator
import os

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

def shannon_entropy(inputfasta):
	seqs, names = read_fasta(inputfasta)
	char_mat = []
	for seq in seqs:
		schar = list(seq)
		char_mat.append(schar)
	char_mat = np.asarray(char_mat)
	print char_mat.size
