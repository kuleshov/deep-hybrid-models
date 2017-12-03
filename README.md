Deep Hybrid Models: Bridging Discriminative and Generative Approaches
=====================================================================

This repository contains code accompanying the paper

```
Deep hybrid models: bridging discriminative and generative approaches.
Volodymyr Kuleshov and Stefano Ermon.
Uncertainty in Artificial Intelligence, 2017
```

## Background

In machine learning, models are typically either discriminative or generative.
Discriminative models often attain higher predictive accuracy, 
while the generative models can be more data efficent.

In our paper, we present a way of defining hybrid models using recent advances in latent variable models and variational inference.
In the context of neural networks, our framework gives rise to deep hybrid models. They are most effective in a semi-supervised setting, where they yield improvements over the state of the art on the SVHN dataset.

For more details, see the [paper](http://web.stanford.edu/~kuleshov/papers/uai2017.pdf).

## Installation

The code uses the latest version of Theano and Lasagne.
To install this package, simply clone the git repo:

```
git clone https://github.com/kuleshov/deep-hybrid-models.git;
cd audio-super-res;
```

In the paper, we define two families of deep hybrid models having, respectively, explicit and implicit densities.
We provide code for each in two separate subfolders.

## Explicit-Density Deep Hybrid Models

This subfolder contains code for fully-supervised explicit-density hybrid models 
in which the generative component is an auxiliary deep generative model, and the discriminative component is a convolutional neural network.

We experiment with this approach in the paper, and show that training both methods improves the accuracy on MNIST and SVHN.

To run a model, you may use the `run.py` launch script.

```
python run.py train \
  --dataset mnist \
  --model supervised-hdgm \
  --alg <opt_alg> \
  --n_batch <batch_size> \
  --lr <learning_rate> \
  -e <num_epochs> \
  -l <log_name>
```

Alternatively, you may use the `Makefile` included in the root dir; typing `make train` will start training. There are also several additional parameters that can be configured i

The model will periodically save its weights and report training/validation losses in the logfile.
The strength of each component is adjusted using the `--sup-weight` and `--unsup-weight` flags.

## Implicit-Density Deep Hybrid Models

This subfolder contains code for semi-supervised implicit-density hybrid models 
in which the generative component is GAN trained with adversarially learned inference (Dumoulin et al., 2017), and the discriminative component is a convolutional neural network with pi-regularization (Laine and Aila, 2017).

We train the GAN and the CNN jointly, and apply both generative and discriminative semi-supervised techniques on the same joint model. This improves semi-supervised error rate from 5% to about 4% on SVHN.

We include a script to reproduce our experiments. Our script is based on the Improved GAN code of Salimans et al. and the temporal ensembling (and pi-regularization) code of Laine and Aila.

```
cd implicit-dhm;
python implicit-semisup-dhm-svhn.py
```

## Feedback

Send feedback to [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov).
