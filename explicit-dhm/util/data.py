import sys
import os
import pickle
import tarfile

import scipy

import numpy as np
# from scipy.ndimage import convolve
try:
  from sklearn import datasets
  from sklearn.cross_validation import train_test_split
except ImportError:
  print "Warning: Couldn't load scikit-learn"

# ----------------------------------------------------------------------------

def whiten(X_train, X_valid):
  offset = np.mean(X_train, 0)
  scale = np.std(X_train, 0).clip(min=1)
  X_train = (X_train - offset) / scale
  X_valid = (X_valid - offset) / scale
  return X_train, X_valid

# ----------------------------------------------------------------------------

def load_cifar10():
  """Download and extract the tarball from Alex's website."""
  dest_directory = '.'
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    if sys.version_info[0] == 2:
      from urllib import urlretrieve
    else:
      from urllib.request import urlretrieve

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)  

  def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
      datadict = pickle.load(f)
      X = datadict['data']
      Y = datadict['labels']
      X = X.reshape(10000, 3, 32, 32).astype("float32")
      Y = np.array(Y, dtype=np.uint8)
      return X, Y

  xs, ys = [], []
  for b in range(1,6):
    f = 'cifar-10-batches-py/data_batch_%d' % b
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch('cifar-10-batches-py/test_batch')
  return Xtr, Ytr, Xte, Yte

def load_mnist():
  # We first define a download function, supporting both Python 2 and 3.
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print "Downloading %s" % filename
    urlretrieve(source + filename, filename)

  # We then define functions for loading MNIST images and labels.
  # For convenience, they also download the requested files if needed.
  import gzip

  def load_mnist_images(filename):
    if not os.path.exists(filename):
      download(filename)
      # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    # data = data.reshape(-1, 1, 28, 28)
    data = data.reshape(-1, 784)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

  def load_mnist_labels(filename):
    if not os.path.exists(filename):
      download(filename)
      # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

  # We can now download and read the training and test set images and labels.
  X_train = load_mnist_images('train-images-idx3-ubyte.gz')
  y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
  X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
  y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

  y_train = y_train.astype('int32')
  y_test = y_test.astype('int32')

  # We reserve the last 10000 training examples for validation.
  X_train, X_val = X_train[:-10000], X_train[-10000:]
  y_train, y_val = y_train[:-10000], y_train[-10000:]

  # We just return all the arrays in order, as expected in main().
  # (It doesn't matter how we do this as long as we can read them again.)
  return X_train, y_train, X_val, y_val, X_test, y_test

def load_svhn():
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source="https://github.com/smlaine2/tempens/raw/master/data/svhn/"):
    print "Downloading %s" % filename
    urlretrieve(source + filename, filename)

  import cPickle
  def load_svhn_files(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    images = []
    labels = []
    for fn in filenames:
        if not os.path.isfile(fn): download(fn)
        with open(fn, 'rb') as f:
          X, y = cPickle.load(f)
        images.append(np.asarray(X, dtype='float32') / np.float32(255))
        labels.append(np.asarray(y, dtype='uint8'))
    return np.concatenate(images), np.concatenate(labels)

  X_train, y_train = load_svhn_files(['train_%d.pkl' % i for i in (1, 2, 3)])
  X_test, y_test = load_svhn_files('test.pkl')

  return X_train, y_train, X_test, y_test

def load_cadd(datadir):
  X_train_npz = np.load(datadir + '/training.X.npz')
  X_val_npz   = np.load(datadir + '/validation.X.npz')
  X_test_npz  = np.load(datadir + '/testing.X.npz')
  X_clinvar_npz  = np.load(datadir + '/ClinVar_ESP.X.npz')

  y_train = np.load(datadir + '/training.y.npy').astype('int32')
  y_val   = np.load(datadir + '/validation.y.npy').astype('int32')
  y_test  = np.load(datadir + '/testing.y.npy').astype('int32')
  y_clinvar = np.load(datadir + '/ClinVar_ESP.y.npy').astype('int32')

  X_train = scipy.sparse.csr_matrix((X_train_npz['data'], X_train_npz['indices'], X_train_npz['indptr']), dtype='float32')
  X_val = scipy.sparse.csr_matrix((X_val_npz['data'], X_val_npz['indices'], X_val_npz['indptr']), dtype='float32')
  X_test = scipy.sparse.csr_matrix((X_test_npz['data'], X_test_npz['indices'], X_test_npz['indptr']), dtype='float32')
  X_clinvar = scipy.sparse.csr_matrix((X_clinvar_npz['data'], X_clinvar_npz['indices'], X_clinvar_npz['indptr']), dtype='float32')

  print X_train.shape
  print X_val.shape

  # X_train = X_train[:10000]
  # X_val = X_val[:10000]
  # y_train = y_train[:10000]
  # y_val = y_val[:10000]

  X_train_tot = scipy.sparse.vstack((X_train, X_val))
  print X_train_tot.shape
  y_train_tot = np.concatenate((y_train, y_val))

  print X_train_tot.shape
  print y_train_tot.shape
  print X_test.shape
  print y_test.shape
  print X_clinvar.shape

  X_clinvar = X_clinvar.todense()

  return X_train_tot, y_train_tot, X_test, y_test, X_clinvar, y_clinvar

def load_augmented_mnist(datadir):
  X_train = pickle.load(open(datadir+'/x.train.pkl')).astype('float32')
  Z_train = pickle.load(open(datadir+'/z.train.pkl')).astype('float32')
  y_train = pickle.load(open(datadir+'/y.train.pkl')).astype('int32')
  
  X_val = pickle.load(open(datadir+'/x.val.pkl')).astype('float32')
  Z_val = pickle.load(open(datadir+'/z.val.pkl')).astype('float32')
  y_val = pickle.load(open(datadir+'/y.val.pkl')).astype('int32')
  
  X_test = pickle.load(open(datadir+'/x.test.pkl')).astype('float32')
  Z_test = pickle.load(open(datadir+'/z.test.pkl')).astype('float32')
  y_test = pickle.load(open(datadir+'/y.test.pkl')).astype('int32')

  X_full_train = np.hstack((X_train, Z_train))
  X_full_test = np.hstack((X_test, Z_test))
  X_full_val = np.hstack((X_val, Z_val))

  X_full_train, X_full_val = X_full_train[:-10000], X_full_train[-10000:]
  y_train, y_val = y_train[:-10000], y_train[-10000:]

  def undo_onehot(y):
    y_out = np.zeros(len(y)).astype('int32')
    for k in range(10):
      y_out[y[:,k]==1]=k
    return y_out

  y_train = undo_onehot(y_train)
  y_val = undo_onehot(y_val)
  y_test = undo_onehot(y_test)

  # print X_train.shape
  # print X_val.shape
  # print X_test.shape

  return X_full_train, y_train, X_full_val, y_val, X_full_test, y_test

# ----------------------------------------------------------------------------
# semisupervised

def split_semisup_simple(X, y, n_lbl):
  n_tot = len(X)
  idx = np.random.permutation(n_tot)
  
  X_lbl = X[idx[:n_lbl]].copy()
  X_unl = X[idx[n_lbl:]].copy()
  y_lbl = y[idx[:n_lbl]].copy()
  y_unl = y[idx[n_lbl:]].copy()

  return X_lbl, y_lbl, X_unl, y_unl

def create_semi_supervised(xy, n_labeled):
    """
    Divide the dataset into labeled and unlabeled data.
    :param xy: The training set of the mnist data.
    :param n_labeled: The number of labeled data points.
    :param rng: NumPy random generator.
    :return: labeled x, labeled y, unlabeled x, unlabeled y.
    """
    x, y = xy
    n_classes = int(np.max(y) + 1)

    def _split_by_class(x, y, n_c):
        result_x = [0] * n_c
        result_y = [0] * n_c
        for i in range(n_c):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y

    x, y = _split_by_class(x, y, n_classes)

    if n_labeled % n_classes != 0:
        raise "n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)"
    n_labels_per_class = n_labeled / n_classes
    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes
    for i in range(n_classes):
        idx = range(x[i].shape[0])
        np.random.shuffle(idx)
        x_labeled[i] = x[i][idx[:n_labels_per_class]]
        y_labeled[i] = y[i][idx[:n_labels_per_class]]
        x_unlabeled[i] = x[i]
        y_unlabeled[i] = y[i]
    return np.concatenate(x_labeled), np.concatenate(y_labeled), np.concatenate(x_unlabeled), np.concatenate(y_unlabeled)

# ----------------------------------------------------------------------------
# other

def split_semisup(X, y, n_lbl, n_class):
  n_tot = len(X)
  idx = np.array()
  for k in range(n_class):
    class_idx = np.array([i for i, yi in enumerate(y) if yi == k])
    rnd_idx = np.random.permutation(len(class_idx))
    rnd_class_idx = class_idx[rnd_idx[:n_lbl/n_class]]
  idx = np.concatenate((idx, rnd_class_idx), axis=1)
  
  X_lbl = X[idx[:n_lbl]].copy()
  X_unl = X[idx[n_lbl:]].copy()
  y_lbl = y[idx[:n_lbl]].copy()
  y_unl = y[idx[n_lbl:]].copy()

  return X_lbl, y_lbl, X_unl, y_unl

def load_digits():
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    X, Y = nudge_dataset(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    n, d2 = X.shape
    d = int(np.sqrt(d2))
    X = X.reshape((n,1,d,d))
    Y = np.array(Y, dtype=np.uint8)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    return X_train, Y_train, X_test, Y_test

def load_noise(n=100,d=5):
  """For debugging"""
  X = np.random.randint(2,size=(n,1,d,d)).astype('float32')
  Y = np.random.randint(2,size=(n,)).astype(np.uint8)

  return X, Y

def load_h5(h5_path):
  """This was untested"""
  import h5py
  # load training data
  with h5py.File(h5_path, 'r') as hf:
    print 'List of arrays in input file:', hf.keys()
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    print 'Shape of X: \n', X.shape
    print 'Shape of Y: \n', Y.shape

    return X, Y

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def prepare_dataset(X_train, y_train, X_test, y_test, aug_translation=0):
  # Whiten input data
  def whiten_norm(x):
    x = x - np.mean(x, axis=(1, 2, 3), keepdims=True)
    x = x / (np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)
    return x

  X_train = whiten_norm(X_train)
  X_test = whiten_norm(X_test)
  
  whitener = ZCA(x=X_train)
  X_train = whitener.apply(X_train)
  X_test = whitener.apply(X_test)

  # Pad according to the amount of jitter we plan to have.

  p = aug_translation
  if p > 0:
      X_train = np.pad(X_train, ((0, 0), (0, 0), (p, p), (p, p)), 'reflect')
      X_test = np.pad(X_test, ((0, 0), (0, 0), (p, p), (p, p)), 'reflect')

  # Random shuffle.
  # indices = np.arange(len(X_train))
  # np.random.shuffle(indices)
  # X_train = X_train[indices]
  # y_train = y_train[indices]

  return X_train, y_train, X_test, y_test    