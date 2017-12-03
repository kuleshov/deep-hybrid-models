import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from supmodel import Model

# ----------------------------------------------------------------------------

class SupervisedCNN(Model):
  def __init__(self, n_dim, n_aug, n_dim_x, n_dim_y, n_chan, n_out, n_superbatch, opt_alg, opt_params):
    self.n_aug=100
    self.model = 'mnist'

    # invoke parent constructor
    Model.__init__(self, n_dim, n_aug, n_dim_x, n_dim_y, n_chan, n_out, n_superbatch, opt_alg, opt_params)
    
  def create_model(self, X, Y, n_dim, n_aug, n_dim_x, n_dim_y, n_out, n_chan=1):
    if self.model == 'mnist':
      return self.create_mnist_model(X, Y, n_dim, n_aug, n_dim_x, n_dim_y, n_out, n_chan)
    elif self.model == 'cifar10':
      return self.create_cifar10_model(X, Y, n_dim_x, n_out, n_chan)
    else:
      raise ValueError('Invalid CNN model type')

  def create_mnist_model(self, X, Y, n_dim, n_aug, n_dim_x, n_dim_y, n_out, n_chan=1):
    X_in = X[:,:n_dim-n_aug]
    X_in = X_in.reshape((-1, n_chan, n_dim_x, n_dim_y))
    # l_in = lasagne.layers.InputLayer(shape=(None, n_dim), input_var=X)
    # l_in = lasagne.layers.ReshapeLayer(l_in, (-1, n_chan, n_dim_x, n_dim_y))
    l_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim_x, n_dim_y), 
                                     input_var=X_in)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    # l_in_drop = l_in

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in_drop, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    l_conv1 = lasagne.layers.MaxPool2DLayer(
        l_conv1, pool_size=(3, 3), stride=(2,2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_conv1, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    l_conv2 = lasagne.layers.MaxPool2DLayer(
        l_conv2, pool_size=(3, 3), stride=(2,2))

    l_conv3 = lasagne.layers.Conv2DLayer(
        l_conv2, num_filters=128, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    l_conv3 = lasagne.layers.MaxPool2DLayer(
        l_conv3, pool_size=(3, 3), stride=(2,2))

    if n_aug > 0:
      X_aug = X[:,n_dim-n_aug:]
      l_aug = self.create_aug_dnn(X_aug, n_aug)
      l_conv3 = lasagne.layers.FlattenLayer(l_conv3)
      l_conv3 = lasagne.layers.ConcatLayer([l_conv3, l_aug])
      # l_conv3 = l_aug

    l_hid = lasagne.layers.DenseLayer(
        l_conv3, num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_hid_drop = lasagne.layers.DropoutLayer(l_hid, p=0.5)
    # l_hid_drop = l_hid

    l_out = lasagne.layers.DenseLayer(
            l_hid_drop, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

  def create_aug_dnn(self, X, n_aug):
    n_hidden = 2000
    l_in = lasagne.layers.InputLayer(shape=(None, n_aug), input_var=X)
    # l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_in_drop = l_in

    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    # l_hid1_drop = l_hid1

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    # l_hid2_drop = l_hid2

    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.5)
    # l_hid3_drop = l_hid3

    return l_hid3_drop

  def create_cifar10_model(self, X, Y, n_dim, n_out, n_chan=1):
    l_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)

    # input layer
    network = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    
    # CCP units
    ccp_num_filters = (64, 128)
    ccp_filter_size = 3
    for num_filters in ccp_num_filters:
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # FC layers
    fc_num_units = (256, 256)
    for num_units in fc_num_units:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=num_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    # output layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    
    return network