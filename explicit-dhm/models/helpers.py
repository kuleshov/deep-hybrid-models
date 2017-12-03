import math
import numpy as np
import scipy

# ----------------------------------------------------------------------------
# iteration

ramp_time = 80
def rampup(epoch):
    if epoch < ramp_time:
        p = max(0.0, float(epoch)) / float(ramp_time)
        p = 1.0 - p
        return np.exp(-p*p*5.0)
    else:
        return 1.0    

def rampdown(epoch, num_epochs, rampdown_length):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0        

class MinibatchIndexIterator():
  def __init__(self, n_inputs, batchsize):
    self.batchsize = batchsize
    self.n_batches = n_inputs / batchsize
    self.idx = 0

  def next(self):
    start_idx = self.idx * self.batchsize
    end_idx = (self.idx+1) * self.batchsize
    self.idx += 1
    if self.idx >= self.n_batches:
      self.idx = 0
    return start_idx, end_idx

def iterate_minibatch_idx(n_inputs, batchsize,):
  for start_idx in range(0, n_inputs - batchsize + 1, batchsize):
    yield start_idx, min(start_idx + batchsize, n_inputs)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert inputs.shape[0] == targets.shape[0]
  if shuffle:
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
  for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
    if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
    else:
        excerpt = slice(start_idx, start_idx + batchsize)

    out_x, out_y = inputs[excerpt], targets[excerpt]
    if scipy.sparse.issparse(out_x):
      out_x = out_x.todense()
    if len(out_x.shape) == 2:
      n, d = out_x.shape
      out_x = np.expand_dims(out_x, 1)
      out_x = np.expand_dims(out_x, 2)
    yield out_x, out_y

def random_subbatch(inputs, targets, batchsize):
  assert inputs.shape[0] == targets.shape[0]
  indices = np.arange(inputs.shape[0])
  np.random.shuffle(indices)
  excerpt = indices[:batchsize]
  return inputs[excerpt], targets[excerpt]

# ----------------------------------------------------------------------------
# eval

def evaluate(eval_f, X, Y, n_metrics=2, batchsize=1000):
  tot_metrics, batches = np.zeros(n_metrics,), 0
  for inputs, targets in iterate_minibatches(X, Y, batchsize, shuffle=False):
    metrics = eval_f(inputs, targets)
    tot_metrics = [t+m for t,m in zip(tot_metrics, metrics)]
    batches += 1
  return [t/batches for t in tot_metrics]

def log_metrics(logname, metrics):
  logfile = '%s.log' % logname
  with open(logfile, 'a') as f:
    f.write('\t'.join([str(m) for m in metrics]) + '\n')