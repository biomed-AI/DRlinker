import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""
    def __init__(self, voc, info, max_size=50):
        self.memory = []
        self.voc = voc
        self.info = info
        self.max_size = max_size

    def add_experience(self, experience):
        """Experience should be a list of (smiles, onehot_seq, score, prior_likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                mol = Chem.MolFromSmiles(exp[0])
                if mol:
                    cano = Chem.MolToSmiles(mol)
                    if cano not in smiles:
                        idxs.append(i)
                        smiles.append(cano)
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key = lambda x: x[2], reverse=True)
            self.memory = self.memory[:self.max_size]

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory)<n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[2]+0.00001 for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            onehot_seqs = [x[1] for x in sample]
            scores = [x[2] for x in sample]
            prior_likelihoods = [x[3] for x in sample]

        return smiles, pad_sequence(onehot_seqs, batch_first=True, padding_value=self.voc.stoi[PAD_WORD]), np.array(scores), torch.tensor(prior_likelihoods)

    def print_memory(self):
        """Prints the memory."""
        self.info("*" * 80 + "\n")
        self.info("Best score in memory: {:.2f}".format(self.memory[0][2]))
        self.info("Score      SMILES\n")

        for exp in self.memory:
            self.info("{:4.2f}        {}".format(exp[2], exp[0]))

        self.info("\n" + "*" * 80 + "\n")

    def clear_memory(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
