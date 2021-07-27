import numpy as np
import scipy.integrate as integrate
import torch

class these_distributions():
	def __init__(self, mean, vari):
		self.mean = mean
		self.vari = vari
	def normal_dist(self, x):
		"""A normal distribution function created to use with scipy.integral.quad
		"""
		mean=self.mean
		vari=self.vari
		return (1.0/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0))/(2.0*vari))

	def expected_normal_dist(self, x):
		"""A expected value of normal distribution function which created to use with scipy.integral.quad
		"""
		mean=self.mean
		vari=self.vari
		return (x/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0))/(2.0*vari))

###not implemented yet....
def laplace_dist(x, mean=0.0, vari=1.0):
    """ A laplace distribution function to use with scipy.integral.quad
    """
    #In laplace distribution beta is used instead of variance so, the converting is necessary.
    scale = np.sqrt(vari/2.0)
    return (1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))

def expected_laplace_dist(x, mean=0.0, vari=1.0):
    """A expected value of laplace distribution function which created to use with scipy.integral.quad
    """
    scale = np.sqrt(vari/2.0)
    return x*(1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))
#####################################################################################

def MSE_loss(x, x_hat_q):
    """Find the mean square loss between x (orginal signal) and x_hat (quantized signal)
    Args:
        x: the signal without quantization
        x_hat_q: the signal of x after quantization
    Return:
        MSE: mean square loss between x and x_hat_q
    """
    #protech in case of input as tuple and list for using with numpy operation
    x = np.array(x)
    x_hat_q = np.array(x_hat_q)
    assert np.size(x) == np.size(x_hat_q)
    MSE = np.sum((x-x_hat_q)**2)/np.size(x)
    return MSE

def generate_delta(last_repre, last_thre):
	delta = np.zeros(len(last_thre)-1)
	for i in range(len(last_thre)-1):
		delta[i] = last_thre[i+1] - last_thre[i]
	return delta

def LloydMaxLayerQuantize(input_tensor, bits, device, voronoi=None):
	mean = torch.mean(input_tensor).item()
	vari = (torch.std(input_tensor)**2).item()

	if vari < 1e-8:
		#print("Entro in 1!")
		if np.abs(mean) < 1e-5:
			H_numel = bits/2.0 + 1.0
			last_repre = np.arange(-(H_numel-1)*0.0001, H_numel*0.0001 + 1e-5, 0.0001)
			for i in range(len(last_repre)):
				if np.abs(last_repre[i])< 0.00001:
					last_repre[i] = 0.0
			x_hat_q = torch.zeros(input_tensor.shape, device = device)
			last_thre = LloydMaxQuantizer.threshold(last_repre)

			return x_hat_q, (last_repre, last_thre, generate_delta(last_repre, last_thre))
		else:
			print(mean)
			print(input_tensor)
			print("Not implemented")
			error()
	if input_tensor.numel() <= bits:
		#print("Entro in 2!")
		delta = (input_tensor.max().item() - input_tensor.min().item())/(bits-1.0)
		last_repre = np.arange(input_tensor.min().item(), input_tensor.max().item() + 1e-5, delta)
		last_repre[0:input_tensor.numel()] = input_tensor.cpu().reshape(-1)
		last_repre = np.sort(last_repre)
		last_repre = np.insert(last_repre, 0, -abs(last_repre[0]-last_repre[1]) + last_repre[0])
		last_repre = np.append(last_repre, abs(last_repre[-2]-last_repre[-1])+last_repre[-1])

		last_thre = LloydMaxQuantizer.threshold(last_repre)
		return input_tensor, (last_repre, last_thre, generate_delta(last_repre, last_thre))

	MD = these_distributions(mean, vari)
	data_cpu = input_tensor.cpu().numpy()
	orig_dims = data_cpu.shape
	data_cpu = data_cpu.reshape(-1)

	if isinstance(voronoi, (list, tuple, np.ndarray)):
		repre = voronoi[0][1:-1]
		thre = voronoi[1][1:-1]
		#thre = LloydMaxQuantizer.threshold(repre)
		#x_hat_q = LloydMaxQuantizer.quant(data_cpu, thre, repre).reshape(orig_dims)
		#repre = np.insert(repre, 0, abs(repre[0]-repre[1]) + repre[0])
		#repre = np.append(repre, abs(repre[-2]-repre[-1])+repre[-1])
		#return(torch.from_numpy(x_hat_q).to(device), repre)
	else:
		repre = LloydMaxQuantizer.start_repre(data_cpu, bits)
		thre = LloydMaxQuantizer.threshold(repre)

	min_loss = 100000.0
	x_hat_q = LloydMaxQuantizer.quant(data_cpu, thre, repre)
	this_loss = MSE_loss(data_cpu, x_hat_q)
	last_thre = thre
	last_repre = repre
	iterid =0
	while (iterid < 100) and (this_loss < min_loss):
		min_loss = this_loss
		last_thre = thre
		last_repre = repre
		thre = LloydMaxQuantizer.threshold(repre)
		#In case wanting to use with another mean or variance, need to change mean and variance in untils.py file
		repre = LloydMaxQuantizer.represent(thre, MD.expected_normal_dist, MD.normal_dist)
		x_hat_q = LloydMaxQuantizer.quant(data_cpu, thre, repre)
		this_loss = MSE_loss(data_cpu, x_hat_q)
		iterid += 1
		'''
		if (last_repre[0] < 0) and (last_repre[-1] > 0):##force some id to zero
			i = 1
			while last_repre[i] < 0:
				i+= 1
			if np.abs(last_repre[i]) < np.abs(last_repre[i-1]):
				last_repre[i] = 0.0
			else:
				last_repre[i-1] = 0.0
		'''


	x_hat_q = LloydMaxQuantizer.quant(data_cpu, last_thre, last_repre).reshape(orig_dims)
	#add boundaries to last thresholds
	last_thre = np.insert(last_thre, 0, data_cpu.min()-0.01)
	last_thre = np.append(last_thre, data_cpu.max()+0.01 )
	#add prohibited bins
	last_repre = np.insert(last_repre, 0, -abs(last_repre[0]-last_repre[1]) + data_cpu.min()-0.01)
	last_repre = np.append(last_repre, abs(last_repre[-2]-last_repre[-1])+data_cpu.max()+0.01)
	#thre = [0.5*(repre[0]+repre[1]) - thre[0], thre, 0.5*(repre[-2]+repre[-1])+thre[-1]]
	return torch.from_numpy(x_hat_q).to(device), (last_repre, last_thre, generate_delta(last_repre, last_thre))

class LloydMaxQuantizer(object):
	"""A class for iterative Lloyd Max quantizer.
	This quantizer is created to minimize amount SNR between the orginal signal
	and quantized signal.
	"""
	@staticmethod
	def start_repre(x, bit):
		"""
		Generate representations of each threshold using
		Args:
		    x: input signal for
		    bit: amount of bit
		Return:
		    threshold:
		"""
		assert isinstance(bit, int)
		x = np.array(x)
		num_repre  = bit#np.power(2,bit)##to allow zero bin
		step = (np.max(x)-np.min(x))/(num_repre)
		repre = np.arange(np.min(x) + step / 2.0, np.max(x), step)
		'''
		middle_point = np.mean(x)
		repre = np.array([])
		for i in range(int(num_repre/2)):
			repre = np.append(repre, middle_point+(i+1)*step)
			repre = np.insert(repre, 0, middle_point-(i+1)*step)
		'''
		#repre = np.insert(repre, int(np.floor(num_repre/2)), 0)
		return repre

	@staticmethod
	def threshold(repre):
		"""
		"""
		t_q = np.zeros(np.size(repre)-1)
		for i in range(len(repre)-1):
			t_q[i] = 0.5*(repre[i]+repre[i+1])
		return t_q

	@staticmethod
	def represent(thre, expected_dist, dist):
		"""
		"""
		thre = np.array(thre)
		x_hat_q = np.zeros(np.size(thre)+1)
		#prepare for all possible integration range
		thre = np.append(thre, np.inf)
		thre = np.insert(thre, 0, -np.inf)

		for i in range(len(thre)-1):
			x_hat_q[i] = integrate.quad(expected_dist, thre[i], thre[i+1])[0]/(integrate.quad(dist,thre[i],thre[i+1])[0]+1e-10)
		return x_hat_q

	@staticmethod
	def represent_discrete(thre, x):
		"""
		"""
		thre = np.array(thre)
		x_hat_q = np.zeros(np.size(thre)+1)
		#prepare for all possible integration range---inf might be extreme
		thre = np.append(thre, np.inf)
		thre = np.insert(thre, 0, -np.inf)

		for i in range(len(thre)-1):
			##select all the parameters falling in range (thre[i], thre[i+1])
			mask = (x > thre[i]) & (x <= thre[i+1])
			bin_x = x[mask == True]
			#print(bin_x.numel())
			x_hat_q[i] = torch.mean(bin_x).item() ##here possible to insert S
			if np.isnan(x_hat_q[i]):##no values dropping here...
				x_hat_q[i] = (thre[i] + thre[i+1]) / 2.0
			#print(x_hat_q[i])
		return x_hat_q

	@staticmethod
	def quant(x, thre, repre):
		"""Quantization operation.
		"""
		thre = np.append(thre, np.inf)
		thre = np.insert(thre, 0, -np.inf)
		x_hat_q = np.zeros(np.shape(x))
		for i in range(len(thre)-1):
			if i == 0:
				x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]),
				                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
			elif i == range(len(thre))[-1]-1:
				x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]),
				                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
			else:
				x_hat_q = np.where(np.logical_and(x > thre[i], x < thre[i+1]),
				                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
		return x_hat_q


class LloydMaxQuantizer_Srelaxed(object):
	"""A class for iterative Lloyd Max quantizer.
	This quantizer is created to minimize amount SNR between the orginal signal
	and quantized signal.
	"""

	@staticmethod
	def start_repre(input_tensor, bits):
		###now we initialize repre optimizing on gaussian prior
		mean = torch.mean(input_tensor).item()
		vari = (torch.std(input_tensor)**2).item()
		MD = these_distributions(mean, vari)
		data_cpu = input_tensor.cpu().numpy()
		orig_dims = data_cpu.shape
		data_cpu = data_cpu.reshape(-1)

		repre = LloydMaxQuantizer_Srelaxed.start_repre_uniform(data_cpu, bits)
		min_loss = 1.0
		for i in range(50):
			thre = LloydMaxQuantizer_Srelaxed.threshold(repre)
			#In case wanting to use with another mean or variance, need to change mean and variance in untils.py file
			repre = LloydMaxQuantizer_Srelaxed.represent_distdiff(thre, MD.expected_normal_dist, MD.normal_dist)
		return repre

	@staticmethod
	def start_repre_uniform(x, bit):
		"""
		Generate representations of each threshold using
		Args:
		    x: input signal for
		    bit: amount of bit
		Return:
		    threshold:
		"""
		assert isinstance(bit, int)
		x = np.array(x)
		num_repre  = np.power(2,bit)
		step = (np.max(x)-np.min(x))/num_repre

		middle_point = np.mean(x)
		repre = np.array([])
		for i in range(int(num_repre/2)):
			repre = np.append(repre, middle_point+(i+1)*step)
			repre = np.insert(repre, 0, middle_point-(i+1)*step)
		return repre

	@staticmethod##requires integrate hence runs on CPU only!
	def represent_distdiff(thre, expected_dist, dist):
		"""
		"""
		thre = np.array(thre)
		x_hat_q = np.zeros(np.size(thre)+1)
		#prepare for all possible integration range
		thre = np.append(thre, np.inf)
		thre = np.insert(thre, 0, -np.inf)

		for i in range(len(thre)-1):
			x_hat_q[i] = integrate.quad(expected_dist, thre[i], thre[i+1])[0]/(integrate.quad(dist,thre[i],thre[i+1])[0])
		return x_hat_q

	@staticmethod
	def threshold(repre):
		"""
		"""
		t_q = np.zeros(np.size(repre)-1)
		for i in range(len(repre)-1):
			t_q[i] = 0.5*(repre[i]+repre[i+1])
		return t_q

	@staticmethod
	def represent_discrete(thre, x):
		"""
		"""
		thre = np.array(thre)
		x_hat_q = np.zeros(np.size(thre)+1)
		#prepare for all possible integration range---inf might be extreme
		thre = np.append(thre, np.inf)
		thre = np.insert(thre, 0, -np.inf)

		for i in range(len(thre)-1):
			##select all the parameters falling in range (thre[i], thre[i+1])
			mask = (x > thre[i]) & (x <= thre[i+1])
			bin_x = x[mask == True]
			#print(bin_x.numel())
			x_hat_q[i] = torch.mean(bin_x).item() ##here possible to insert S
			if np.isnan(x_hat_q[i]):##no values dropping here...
				x_hat_q[i] = (thre[i] + thre[i+1]) / 2.0
			#print(x_hat_q[i])
		return x_hat_q

	@staticmethod
	def quantize(x, thre, repre, device):
		"""Quantization operation.
		"""
		thre = np.append(thre, np.inf)
		thre = np.insert(thre, 0, -np.inf)
		###for the moment, on CPU only
		x_np = x.cpu().numpy()
		x_hat_q = np.zeros(np.shape(x))
		for i in range(len(thre)-1):
			if i == 0:
				x_hat_q = np.where(np.logical_and(x_np > thre[i], x_np <= thre[i+1]),
				                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
			elif i == range(len(thre))[-1]-1:
				x_hat_q = np.where(np.logical_and(x_np > thre[i], x_np <= thre[i+1]),
				                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
			else:
				x_hat_q = np.where(np.logical_and(x_np > thre[i], x_np < thre[i+1]),
				                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
		return torch.from_numpy(x_hat_q).to(device)
