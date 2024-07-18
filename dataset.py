import os
import torch
import itertools, os
import numpy as np
from progress.bar import Bar

from Utils.EventSeq import EventSeq
from Utils.ControlSeq import ControlSeq
from Utils import utils
import config

import torch

# pylint: disable=E1101
# pylint: disable=W0101

class Dataset:
    def __init__(self, root, verbose=False):
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, ['.data'])

        self.root = root
        self.samples = []
        self.seqlens = []
        self.samples2 = []
        self.seqlens2 = []

        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            eventseq, eventseq2, controlseq, controlseq2 = torch.load(path)
            controlseq = ControlSeq.recover_compressed_array(controlseq)
            controlseq2 = ControlSeq.recover_compressed_array(controlseq2)
            assert len(eventseq) == len(controlseq)
            assert len(eventseq2) == len(controlseq2)
            self.samples.append((eventseq, controlseq))
            self.seqlens.append(len(eventseq))
            self.samples2.append((eventseq2,controlseq2))
            self.seqlens2.append(len(eventseq2))

        self.avglen = np.mean(self.seqlens)
        self.avglen2 = np.mean(self.seqlens2)
    
    def batches(self, batch_size, window_size, stride_size):
        indeces = [(i, range(j, j + window_size))
                   for i, seqlen in enumerate(self.seqlens)
                   for j in range(0, seqlen - window_size, stride_size)]
        while True:
            eventseq_batch = []
            controlseq_batch = []
            eventseq_batch2 = []
            controlseq_batch2 = []
            n = 0
            for ii in np.random.permutation(len(indeces)):
                i, r = indeces[ii]

                eventseq, controlseq = self.samples[i]
                eventseq2, controlseq2 = self.samples2[i]

                eventseq = eventseq[r.start:r.stop]
                eventseq2 = eventseq2[r.start:r.stop]

                controlseq = controlseq[r.start:r.stop]
                controlseq2 = controlseq2[r.start:r.stop]

                eventseq_batch.append(eventseq)
                controlseq_batch.append(controlseq)
                eventseq_batch2.append(eventseq2)
                controlseq_batch2.append(controlseq2)

                n += 1
                if n == batch_size:
                    yield (np.stack(eventseq_batch, axis=1),
                           np.stack(controlseq_batch, axis=1),
                           np.stack(eventseq_batch, axis=1),
                           np.stack(controlseq_batch, axis=1))
                    eventseq_batch.clear()
                    controlseq_batch.clear()
                    eventseq_batch2.clear()
                    controlseq_batch2.clear()
                    n = 0
    
    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')
