import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import scipy
import scipy.misc
import svhn_data

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--count', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='.')
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load SVHN data
def rescale(mat):
    return np.transpose(np.cast[th.config.floatX]((-127.5 + mat)/127.5),(3,2,0,1))

trainx, trainy = svhn_data.load(args.data_dir,'train')
testx, testy = svhn_data.load(args.data_dir,'test')
trainx = rescale(trainx)
testx = rescale(testx)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(np.ceil(float(testx.shape[0])/args.batch_size))

# input layers
noise_dim = (args.batch_size, 100)
noise = theano_rng.uniform(size=noise_dim)
x_input = ll.InputLayer(shape=(None, 3, 32, 32))
z_input = ll.InputLayer(shape=noise_dim, input_var=noise)

# specify generative model
gen_layers = [z_input]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu, name='g1'), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='g2'), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='g3'), g=None)) # 8 -> 16
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh, name='g4'), train_g=True, init_stdv=0.1)) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1])

genz_layers = [x_input]
genz_layers.append(dnn.Conv2DDNNLayer(genz_layers[-1], 128, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='gz1'))
genz_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(genz_layers[-1], 256, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='gz2'), g=None))
genz_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(genz_layers[-1], 512, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='gz2'), g=None))
genz_layers.append(nn.batch_norm(ll.NINLayer(genz_layers[-1], num_units=512, W=Normal(0.05), nonlinearity=nn.lrelu, name='gz5'), g=None))
genz_layers.append(ll.GlobalPoolLayer(genz_layers[-1], name='gz6'))
genz_layers.append(ll.DenseLayer(genz_layers[-1], num_units=100, W=Normal(0.05), nonlinearity=lasagne.nonlinearities.sigmoid, name='gz7'))


# specify discriminative model

# for z
discz_layers = [z_input]
discz_layers.append(ll.DropoutLayer(discz_layers[-1], p=0.2))
discz_layers.append(ll.DenseLayer(discz_layers[-1], 500, W=Normal(0.05), nonlinearity=nn.lrelu, name='dz1'))
discz_layers.append(ll.DropoutLayer(discz_layers[-1], p=0.5))
discz_layers.append(ll.DenseLayer(discz_layers[-1], 250, W=Normal(0.05), nonlinearity=nn.lrelu, name='dz2'))
discz_layers.append(ll.DropoutLayer(discz_layers[-1], p=0.5))

# for x
disc_layers = [x_input]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dcnn1')))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dcnn2')))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='dcnn3')))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dcnn4')))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dcnn5')))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='dcnn6')))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 256, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu, name='dcnn7')))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu, name='dnin1')))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=64, W=Normal(0.05), nonlinearity=nn.lrelu, name='dnin2')))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))

# combine with disc_z
disc_layers.append(ll.ConcatLayer([disc_layers[-1], discz_layers[-1]]))

# finalize
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
disc_params = ll.get_all_params(disc_layers, trainable=True)

# costs
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()
temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(genz_layers[-1], {x_input: x_lab}, deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], {x_input: x_lab}, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers+discz_layers+genz_layers for u in getattr(l,'init_updates',[])]

genz_lab = ll.get_output(genz_layers[-1], {x_input: x_lab})
genz_unl = ll.get_output(genz_layers[-1], {x_input: x_unl})

output_before_softmax_lab = ll.get_output(disc_layers[-1], {x_input: x_lab, z_input: genz_lab}, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], {x_input: x_unl, z_input: genz_unl}, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], {x_input: gen_dat}, deterministic=False)

# copies for stability regularizer
unsup_weight_var = T.scalar('unsup_weight')
x_lab2 = T.tensor4()
x_unl2 = T.tensor4()
genz_lab2 = ll.get_output(genz_layers[-1], {x_input: x_lab2})
genz_unl2 = ll.get_output(genz_layers[-1], {x_input: x_unl2})
output_before_softmax_lab2 = ll.get_output(disc_layers[-1], {x_input: x_lab2, z_input: genz_lab2}, deterministic=False)
output_before_softmax_unl2 = ll.get_output(disc_layers[-1], {x_input: x_unl2, z_input: genz_unl2}, deterministic=False)

l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
l_gen = nn.log_sum_exp(output_before_softmax_gen)
loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], {x_input: x_lab, z_input: genz_lab}, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)

# create disc loss
loss_stab_lab = T.mean(lasagne.objectives.squared_error(
    T.nnet.softmax(output_before_softmax_lab), T.nnet.softmax(output_before_softmax_lab2)))
loss_stab_unl = T.mean(lasagne.objectives.squared_error(
    T.nnet.softmax(output_before_softmax_unl), T.nnet.softmax(output_before_softmax_unl2)))
loss_disc = (loss_lab + unsup_weight_var * loss_stab_lab) \
          + args.unlabeled_weight*(loss_unl + unsup_weight_var * loss_stab_unl)

disc_param_updates = nn.adam_updates(disc_params, loss_disc, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)

train_batch_disc = th.function(inputs=[x_lab,x_lab2,labels,x_unl,x_unl2,lr,unsup_weight_var], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
test_batch = th.function(inputs=[x_lab], outputs=output_before_softmax, givens=disc_avg_givens)
samplefun = th.function(inputs=[],outputs=gen_dat)

# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-2], {x_input: x_unl, z_input: genz_unl}, deterministic=False)
output_gen = ll.get_output(disc_layers[-2], {x_input: gen_dat}, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(abs(m1-m2))

gen_params = ll.get_all_params(gen_layers + genz_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=None, updates=gen_param_updates)

# select labeled data
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

def augment(X, p=2):
    X_aug = np.zeros(X.shape, dtype='float32')
    X = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), 'reflect')
    for i in range(len(X)):
        ofs0 = np.random.randint(-p, p + 1) + p
        ofs1 = np.random.randint(-p, p + 1) + p
        X_aug[i,:,:,:] = X[i, :, ofs0:ofs0+32, ofs1:ofs1+32]
    return X_aug

ramp_time = 30
def rampup(epoch):
    if epoch < ramp_time:
        p = max(0.0, float(epoch)) / float(ramp_time)
        p = 1.0 - p
        return np.exp(-p*p*5.0)
    else:
        return 1.0    

# //////////// perform training //////////////
for epoch in range(900):
    begin = time.time()

    # set unsupervised weight
    scaled_unsup_weight_max = 100
    rampup_value = rampup(epoch)
    unsup_weight = rampup_value * scaled_unsup_weight_max
    unsup_weight = np.cast[th.config.floatX](unsup_weight)

    # set learning rate
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/300., 1.))
    # lr = np.cast[th.config.floatX](args.learning_rate * rampup_value * np.minimum(3. - epoch/300., 1.))

    # construct randomly permuted minibatches
    trainx = []
    trainxa = []
    trainxb = []
    trainx2 = []
    trainx2 = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        # trainxa.append(augment(txs[inds]))
        # trainxb.append(augment(txs[inds]))
        trainxa.append((txs[inds]))
        trainxb.append((txs[inds]))
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainxa = np.concatenate(trainxa, axis=0)
    trainxb = np.concatenate(trainxb, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    
    # trainx_unla = augment(trainx_unl)
    # trainx_unlb = augment(trainx_unl)
    trainx_unla = (trainx_unl)
    trainx_unlb = (trainx_unl)

    # print 'making samples...'
    # for i in range(5):
    #     scipy.misc.imsave('%da.png' % i, trainx_unla[i,0,:,:])
    #     scipy.misc.imsave('%db.png' % i, trainx_unlb[i,0,:,:])
    #     scipy.misc.imsave('%dc.png' % i, trainxa[i,0,:,:])
    #     scipy.misc.imsave('%dd.png' % i, trainxb[i,0,:,:])

    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]
    
    if epoch==0:
        print(trainx.shape)
        init_param(trainx[:100]) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ran_from = t*args.batch_size
        ran_to = (t+1)*args.batch_size
        ll, lu, te = train_batch_disc(trainxa[ran_from:ran_to],
                                      trainxb[ran_from:ran_to],
                                      trainy[ran_from:ran_to],
                                      trainx_unla[ran_from:ran_to],
                                      trainx_unlb[ran_from:ran_to],
                                      lr, unsup_weight)
        loss_lab += ll
        loss_unl += lu
        train_err += te
        
        train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train

    # test
    test_pred = np.zeros((len(testy),10), dtype=th.config.floatX)
    for t in range(nr_batches_test):
        last_ind = np.minimum((t+1)*args.batch_size, len(testy))
        first_ind = last_ind - args.batch_size
        test_pred[first_ind:last_ind] = test_batch(testx[first_ind:last_ind])
    test_err = np.mean(np.argmax(test_pred,axis=1) != testy)

    expname = 'ali-pimodel-%.4fuw-trueali-largerdz-noaugment-seed%d' % (args.unlabeled_weight, args.seed)
    out_str = "Experiment %s, Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (expname, epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err)
    print(out_str)
    sys.stdout.flush()
    with open(expname + '.log', 'a') as f:
        f.write(out_str + '\n')

    # sample
    imgs = samplefun()
    imgs = np.transpose(imgs[:100,], (0, 2, 3, 1))
    imgs = [imgs[i, :, :, :] for i in range(100)]
    rows = []
    for i in range(10):
        rows.append(np.concatenate(imgs[i::10], 1))
    imgs = np.concatenate(rows, 0)
    scipy.misc.imsave("svhn_sample_feature_match.png", imgs)

    # save params
    np.savez('%s.disc_params.npz' % expname,*[p.get_value() for p in disc_params])
    np.savez('%s.gen_params.npz' % expname,*[p.get_value() for p in gen_params])



