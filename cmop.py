import matplotlib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize

matplotlib.use('pdf')
import os
import tqdm
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.python.framework import function
from generator import Vgg19
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
w_init = tf.contrib.layers.xavier_initializer

def data_iterator():
    while True:
        idxs = np.arange(0, len(data))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(data), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = data[cur_idxs]
            if len(images_batch) < batch_size:
                break

            yield images_batch


@function.Defun(tf.float32, tf.float32, tf.float32, tf.float32)
def DoublySNGrad(logits, epsilon, dprev, dpout):
    prob = 1.0 / (1 + tf.exp(-logits))
    # yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    # {-1, 1} coding
    # yout = tf.sign(prob - epsilon)

    # unbiased
    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon


@function.Defun(tf.float32, tf.float32, grad_func=DoublySNGrad)
def DoublySN(logits, epsilon):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = tf.sign(prob - epsilon)
    # yout = tf.sign(prob - epsilon)
    return yout, prob


def score_node_classification(features, z, p_labeled=0.02, n_repeat=10, norm=False):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm

    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    lrcv = LogisticRegressionCV()

    if norm:
        features = normalize(features)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])

        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')

        trace.append((f1_micro, f1_macro))

    return np.array(trace).mean(0)

def inference(input):
    with tf.variable_scope("enc"):
        W_mu = tf.get_variable(name='W_mu', shape=[300, 512], dtype=tf.float32, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[512], dtype=tf.float32, initializer=w_init())
        h1 = tf.matmul(input, W_mu) + b_mu
        h1 = tf.nn.tanh(h1)

        W_mu1= tf.get_variable(name='W_mu1', shape=[512, 16], dtype=tf.float32, initializer=w_init())
        b_mu1 = tf.get_variable(name='b_mu1', shape=[16], dtype=tf.float32, initializer=w_init())
        h2 = tf.matmul(h1, W_mu1) + b_mu1
        h2 = tf.nn.tanh(h2)


        hepsilon = tf.ones(shape=tf.shape(h2), dtype=tf.float32) * .5
        yout, pout = DoublySN(h2, hepsilon)
        yout = tf.reshape(yout, [-1,16])
        # yout = h2


    with tf.variable_scope("dec"):
        W_mu2 = tf.get_variable(name='W_mu2', shape=[16, 512], dtype=tf.float32, initializer=w_init())
        b_mu2 = tf.get_variable(name='b_mu2', shape=[512], dtype=tf.float32, initializer=w_init())
        h3 = tf.matmul(yout, W_mu2) + b_mu2
        h3 = tf.nn.tanh(h3)
        W_mu3 = tf.get_variable(name='W_mu3', shape=[512, 300], dtype=tf.float32, initializer=w_init())
        b_mu3 = tf.get_variable(name='b_mu3', shape=[300], dtype=tf.float32, initializer=w_init())
        h4 = tf.matmul(h3, W_mu3) + b_mu3
        h4 = tf.nn.tanh(h4)
    return yout, h4


def loss(input, recon):
    MSE_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(input - recon),1))
    Mean_sse = tf.reduce_mean(tf.square(input - recon))
    return MSE_loss1, Mean_sse


batch_size = 1000
graph = tf.Graph()

with graph.as_default():
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    opt_E = tf.train.AdamOptimizer(0.1, epsilon=1.0)
with graph.as_default():
    input = tf.placeholder(tf.float32, [None, 300])

    with tf.device('/gpu:1'):
        with tf.name_scope('Tower_0') as scope:
            compress_code,reconstruction = inference(input)
            SSE_loss1, msse = loss(input, reconstruction)
            params = tf.trainable_variables()
            E_params = [i for i in params if 'enc' in i.name]
            G_params = [i for i in params if 'dec' in i.name]
            grads_e = opt_E.compute_gradients(SSE_loss1, var_list=E_params + G_params)

with graph.as_default():

    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)


with graph.as_default():
    saver = tf.train.Saver()

    sess = tf.InteractiveSession(graph=graph,
                                 config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    sess.run(tf.global_variables_initializer())
data=np.load('data/glove.6B.300d.npy')[:-40000]
test=data[-40000:]
print data.shape
iter = data_iterator()
epoch = 0
num_epochs = 10000
while epoch < num_epochs:
    train_batch = iter.next()
    _, loss_ms = sess.run([train_E,msse],{input:train_batch})
    print 'epoch: ',epoch, 'loss: ',loss_ms
    epoch +=1
nn = []
D = len(test) /batch_size
for i in xrange(D):
    train_batch = test[i*batch_size:(i+1)*batch_size,:]
    cc = sess.run(reconstruction, {input: train_batch})
    nn.append(cc)

print "start classify!"
nn=np.row_stack(nn)
# f,h = score_node_classification(nn,dump['label'])
# print f,h
#
# f,h = score_node_classification(dump['feat'],dump['label'])
# print f,h
print nn.shape,test.shape
distances=[]
for i in xrange(D):
    distances.extend(np.linalg.norm(nn[i*batch_size:(i+1)*batch_size,:] - test[i*batch_size:(i+1)*batch_size,:],axis=1).tolist())
print np.mean(distances)

# np.savez('origal.npz',data=dump['feat'])
# np.savez('comp.npz',data=nn)