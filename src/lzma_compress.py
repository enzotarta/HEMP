import py7zr
import os
import torch
from copy import deepcopy
import numpy as np

def lzma_compress(target, source):
	arch7 = py7zr.SevenZipFile(target, 'w')
	arch7.writeall(source)
	arch7.close()
	return os.path.getsize(target)

def generate_sequence_tocompress(q_model, voronoi, mode = 0):
	qmodel_sd = deepcopy(q_model.state_dict())
	sequence = []
	for k in qmodel_sd:
		if k in voronoi.keys():
			if voronoi[k] != None:
				for i in np.arange(1, len(voronoi[k][0])-1):
					qmodel_sd[k] = qmodel_sd[k]*(qmodel_sd[k]!=voronoi[k][0][i]) + (qmodel_sd[k]==voronoi[k][0][i])*(i-1)
				this_sequence = qmodel_sd[k]
				if len(this_sequence.shape) > 2:##then it is conv
					if mode == 1:
						this_sequence = this_sequence.transpose(2,3)##entropy computed through neurons
					elif mode == 2:
						this_sequence = this_sequence.transpose(1,3)##entropy computed through neurons
					elif mode == 3:
						this_sequence = this_sequence.transpose(0,3)##entropy computed through neurons
				elif (len(this_sequence.shape) == 2) and (mode == 3):
					this_sequence = this_sequence.transpose(0,1)
				this_sequence = this_sequence.reshape(-1).cpu()
				if len(sequence)==0:
					sequence = this_sequence.type(torch.uint8)
				else:
					sequence = torch.cat((sequence, this_sequence.type(torch.uint8)), 0)
	return sequence