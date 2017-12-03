import argparse
from util import data

# ----------------------------------------------------------------------------

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train

  train_parser = subparsers.add_parser('train', help='Train model')
  train_parser.set_defaults(func=train)

  train_parser.add_argument('--dataset', default='mnist')
  train_parser.add_argument('--model', default='softmax')
  train_parser.add_argument('--reload')
  train_parser.add_argument('-e', '--epochs', type=int, default=10)
  train_parser.add_argument('-l', '--logname', default='mnist-run')
  train_parser.add_argument('--alg', default='adam')
  train_parser.add_argument('--lr', type=float, default=1e-3)
  train_parser.add_argument('--b1', type=float, default=0.9)
  train_parser.add_argument('--b2', type=float, default=0.999)
  train_parser.add_argument('--n_batch', type=int, default=128)
  train_parser.add_argument('--n_lbl_batch', type=int, default=1024)
  train_parser.add_argument('--n_superbatch', type=int, default=1280)
  train_parser.add_argument('--n_labeled', type=int, default=1000)
  train_parser.add_argument('--unsup-weight', type=float, default=1.0)
  train_parser.add_argument('--sup-weight', type=float, default=1.0)

  return parser

# ----------------------------------------------------------------------------

def train(args):
  import models
  import numpy as np
  # np.random.seed(1234)

  if args.dataset == 'digits':
    n_dim_x, n_dim_y, n_out, n_channels = 8, 8, 10, 1
    X_train, y_train, X_val, y_val = data.load_digits()
  elif args.dataset == 'mnist':
    # load supservised data
    n_dim, n_aug, n_dim_x, n_dim_y, n_out, n_channels = 784, 0, 28, 28, 10, 1
    X_train, y_train, X_val, y_val, X_test, y_test = data.load_mnist()
    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))
    X_val, y_val = X_test, y_test
  elif args.dataset == 'svhn':
    n_dim, n_aug, n_dim_x, n_dim_y, n_out, n_channels = 3072, 0, 32, 32, 10, 3
    X_train, y_train, X_val, y_val = data.load_svhn()
  elif args.dataset == 'random':
    X_train, y_train = data.load_noise(n=1000, d=n_dim)
  else:
    raise ValueError('Invalid dataset name: %s' % args.dataset)
  print 'dataset loaded.'

  # set up optimization params
  p = { 'lr' : args.lr, 'b1': args.b1, 'b2': args.b2 }

  # X_train_unl = X_train_unl.reshape((-1, n_channels, n_dim_x, n_dim_y))
  # X_test = X_test.reshape((-1, n_channels, n_dim_x, n_dim_y))

  # create model
  if args.model == 'supervised-mlp':
    model = models.SupervisedMLP(n_out=n_out, n_dim=n_dim, n_aug=n_aug, n_dim_x=n_dim_x, n_dim_y=n_dim_y, n_chan=n_channels,
                       n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p)
  elif args.model == 'supervised-hdgm':
    X_train = X_train.reshape(-1, n_channels, n_dim_x, n_dim_y)
    X_val = X_val.reshape(-1, n_channels, n_dim_x, n_dim_y)
    model = models.SupervisedHDGM(n_out=n_out, n_dim=n_dim_x, n_chan=n_channels,
                       n_superbatch=args.n_superbatch, opt_alg=args.alg, opt_params=p,
                       model='bernoulli' if args.dataset=='mnist' else 'gaussian')    
  else:
    raise ValueError('Invalid model')

  if args.reload: model.load(args.reload)

  logname = '%s-%s-%s-%d-%f-%f' % (args.logname, args.dataset, args.model, args.n_labeled, args.sup_weight, args.unsup_weight)
  
  # train model
  model.fit(X_train, y_train, X_val, y_val, 
            n_epoch=args.epochs, n_batch=args.n_batch,
            logname=logname)

def main():
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)

if __name__ == '__main__':
  main()