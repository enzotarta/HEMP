import torch

class S(object):
	def __init__(self, model, device, mom=0.0):
		self.mom = mom
		self.device = device
		self.grad_hist={}
		for n,p in model.named_parameters():
			self.grad_hist[n] = torch.zeros(p.data.shape, device=self.device)
	
	@torch.no_grad()
	def update(self, model):
		for n,p in model.named_parameters():
			self.grad_hist[n] = self.grad_hist[n] * self.mom + p.grad.data
	
	@torch.no_grad()
	def get_Sbar(self, name):
		this_abs = torch.abs(self.grad_hist[name])
		this_max = torch.max(this_abs)
		if this_max == 0:
			return torch.zeros(this_abs.shape, device = self.device)
		this_S = torch.clamp(this_abs / this_max, 0.0, 1.0)
		return 1.0 - this_S
	
	def __del__(self):
		print('Destructor called, Employee deleted.')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def plot_w_dist(model, q_model, voronoi, histpath, epoch):
	q_model_sd = q_model.state_dict()
	for n,p in model.named_parameters():
		this_p = p.data.cpu().numpy().reshape(-1)
		plt.title(n+"   epoch:"+str(epoch))
		plt.xlabel("value")
		plt.ylabel("frequency")
		plt.hist(this_p, bins='auto', alpha=0.5, label="parameter_distribution")
		plt.legend()
		plt.savefig(histpath + n + "_" + str(epoch) + ".png")
		plt.clf()
	plt.close()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res