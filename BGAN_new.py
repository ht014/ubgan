import matplotlib
matplotlib.use('pdf')
import tensorflow as tf
import numpy as np
from generator import Vgg19
import os
import utils
import scipy
import scipy.io as sio
from deconv import deconv2d
import tqdm
import scipy
import scipy.io as sio
import prettytensor as pt
import matplotlib.pyplot as plt
from deconv import deconv2d
import  IPython
import IPython.display
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
from tensorflow.python.framework import function
import myalexnet_forward_newtf as alexnet
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def saveB(name_B):
    a = sio.loadmat('cifar-10.mat')
    dataset = a['data_set']
    test = a['test_data']
    images_ = []
    for i in tqdm.tqdm(range(59000)):
        t = dataset[:, :, :, i]
        image = scipy.misc.imresize(t, [224, 224])
        images_.append(image)
    images_ = np.array(images_)
    test_images = []
    for i in tqdm.tqdm(range(1000)):
        t = test[:, :, :, i]
        image = scipy.misc.imresize(t, [224, 224])
        test_images.append(image)
    test_images = np.array(test_images)
    print 'generate binary codes ....'
    for i in xrange(0, len(images_), batch_size):
        all = images_[i:i + batch_size]
        feature = sess.run([hiden_layer2], \
                           feed_dict= {all_input224: all,train_model: False })
        if i == 0:
            B = feature[0]
        else:
            B = np.concatenate((B, feature[0]), axis=0)

    for i in xrange(0, len(test_images), batch_size):
        all = test_images[i:i + batch_size]
        feature =  sess.run([hiden_layer2], \
                            feed_dict={all_input224: all ,train_model: False})
        if i == 0:
            B_ = feature[0]
        else:
            B_ = np.concatenate((B_, feature[0]), axis=0)
    np.savez(name_B+'.npz', dataset=B, test=B_)
    print 'save done!'


def  data_iterator():
    while True:
        idxs = np.arange(0, len(img64))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(img64), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = img64[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs

def data_iterator_test():
    while True:
        idxs = np.arange(0, len(test_data))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(test_data), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = test_data[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs


from srez_model import *

def generator2( features, channels=3):

    # Upside-down all-convolutional resnet

    mapsize = 3
    res_units = [512, 256, 128,64]  #[256, 128, 96,64]

    old_vars = tf.all_variables()

    # See Arxiv 1603.05027
    model = Model('GEN2', features)

    for ru in range(len(res_units)-1):
        nunits = res_units[ru]

        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)

        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
        model.add_upscale()

        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        # Finalization a la "all convolutional net"
    nunits = res_units[-1]
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()

    # Last layer is sigmoid with no batch normalization
    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
    model.add_sigmoid()

    new_vars = tf.all_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars

def super_resolution( features, channels=3):

    # Upside-down all-convolutional resnet

    mapsize = 3
    res_units = [256, 128, 96,64]

    old_vars = tf.all_variables()

    # See Arxiv 1603.05027
    model = Model('GEN2', features)

    for ru in range(len(res_units) - 1):
        nunits = res_units[ru]

        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)

        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
        # model.add_upscale()

        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

    # Finalization a la "all convolutional net"
    nunits = res_units[-1]
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()

    # Last layer is sigmoid with no batch normalization
    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
    model.add_sigmoid()

    new_vars = tf.all_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars


def encode(disc_input):
    mapsize = 3
    layers = [64, 128, 256, 512]

    old_vars = tf.all_variables()

    model = Model('DIS',  disc_input )

    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0

        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

    # Finalization a la "all convolutional net"
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()
    # print model.get_output().shape
    # raw_input()
    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()
    # print model.get_output().shape
    # raw_input()
    model.add_flatten()
    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    # model.add_dense(1024)
    # model.add_batch_norm()
    # model.add_relu()
    model.add_dense(hidden_size)

    new_vars = tf.all_variables()
    disc_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), disc_vars


def discriminator2(disc_input):
    # Fully convolutional model
    mapsize = 3
    layers = [64, 128, 256, 512]

    old_vars = tf.all_variables()

    model = Model('DIS', disc_input)

    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0
        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

    # Finalization a la "all convolutional net"
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()

    new_vars = tf.all_variables()
    disc_vars = list(set(new_vars) - set(old_vars))
    return model.get_output(), disc_vars

def create_image(im):
    return np.reshape(im,(64,64,3))

test_iterator = data_iterator_test()
def plot_network_output(name_fig):
    real_imag, _1 = test_iterator.next()
    nxt224 = test224[_1]

    recon_x, sup, MSE1, MSE2 = sess.run([ gen_img1, super_img, SSE_loss1,SSE_loss2  ], { all_input224: nxt224 ,
                                        all_input64: real_imag, train_model: True  })
    examples = 14
    recon_x1 = np.squeeze(recon_x)
    recon_x2 = np.squeeze(sup)

    fig, ax = plt.subplots(nrows=3,ncols=examples, figsize=(18,6))
    for i in xrange(examples):
        ax[(0,i)].imshow(create_image(recon_x1[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,i)].set_title(str(MSE1))
        ax[(1, i)].set_title(str(MSE2))
        ax[(1, i)].imshow(create_image(recon_x2[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(2, i)].imshow(create_image(real_imag[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,i)].axis('off')
        ax[(1,i)].axis('off')
        ax[(2, i)].axis('off')
    fig.suptitle('Bottom: inputs | Top: reconstructions')
    plt.savefig(name_fig)
    plt.close()


def plot_network_output_train(name_fig,real_imag,recons,sre_recons,MSE1,MSE2):

    examples = 14
    recon_x = np.squeeze(recons)

    fig, ax = plt.subplots(nrows=3,ncols=examples, figsize=(18,6))
    for i in xrange(examples):
        ax[(0, i)].imshow(create_image(recon_x[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0, i)].set_title(str(MSE1))
        ax[(1, i)].imshow(create_image(sre_recons[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(1, i)].set_title(str(MSE2))
        ax[(2, i)].imshow(create_image(real_imag[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,i)].axis('off')
        ax[(1,i)].axis('off')
        ax[(2, i)].axis('off')

    fig.suptitle('Bottom: inputs | Top: reconstructions')
    plt.savefig(name_fig)
    plt.close()


@function.Defun(tf.float32, tf.float32, tf.float32, tf.float32)
def DoublySNGrad(logits, epsilon, dprev, dpout):
    prob = 1.0 / (1 + tf.exp(-logits))
    #yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    # {-1, 1} coding
    # yout = tf.sign(prob - epsilon)

    # unbiased
    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon

@function.Defun(tf.float32, tf.float32, grad_func=DoublySNGrad)
def DoublySN(logits, epsilon):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon)) #+ 1.0) / 2.0
    # yout = tf.sign(prob - epsilon)
    return yout, prob


def inference(x64,xx224):

        with tf.variable_scope("enc"):
            hiden_layer__,_= encode(x64)
            # yout = tf.nn.tanh(hiden_layer__)
            # before_hiden_layer_= alexnet.create_AlextNet(x64)
            # # print before_hiden_layer_.shape
            #
            # before_hiden_layer = tf.reshape(before_hiden_layer_, [batch_size, -1])
            #
            # hash_weight = tf.Variable(
            #     tf.random_normal([2304, hidden_size], stddev=1.0 / tf.sqrt(float(4096)), dtype=tf.float32),
            #     name='wdecode232')
            # hash_bias = tf.Variable(tf.random_normal([hidden_size], stddev=1.0 / tf.sqrt(float(4096)), dtype=tf.float32),
            #                     name='bias232')
            #
            # hiden_layer = tf.matmul(before_hiden_layer, hash_weight) + hash_bias
            # vgg_net = Vgg19('./vgg19.npy', codelen=hidden_size)
            # vgg_net.build(xx224,train_model)
            # hiden_layer =vgg_net.fc9
            # yout = tf.sign(hiden_layer)
            hepsilon = tf.ones(shape=tf.shape(hiden_layer__), dtype=tf.float32) * .5
            yout, pout = DoublySN(hiden_layer__, hepsilon)
        with tf.variable_scope("gen"):
            wdecode = tf.Variable(tf.random_normal([hidden_size, 8*8*64], stddev=1.0 / tf.sqrt(float(4096)), dtype=tf.float32),name='wdecode')
            baise = tf.Variable(tf.random_normal([8*8*64], stddev=1.0 / tf.sqrt(float(4096)), dtype=tf.float32),name='bias')
            new_space_ = tf.matmul(yout, wdecode) + baise
            new_space =tf.reshape(tf.nn.relu(new_space_),[batch_size,8,8,64])
            gen_img,_ = generator2(new_space)
        with tf.variable_scope('srez'):
            gen_img2, _ = super_resolution(gen_img)
        with tf.variable_scope("dis"):
            dis_fake,_ = discriminator2(gen_img2)
        with tf.variable_scope("dis", reuse=True):
            dis_real, _ = discriminator2(x64)
        # with tf.variable_scope('feat'):
        #     vgg_net = Vgg19('./vgg19.npy', codelen=hidden_size)
        #     real_feats = vgg_net.build(x64)
        # with tf.variable_scope('feat',reuse=True):
        #     fake_feats = vgg_net.build(gen_img2)
        return hiden_layer__,yout,gen_img,dis_fake,dis_real,gen_img2


def loss(x64, gen_img,gen_img2,dis_fake,dis_real,real_hash1,real_hash2,similairy):
    # feat_loss =  0.00001*(tf.nn.l2_loss(real_feats[0]-fake_feats[0]) \
    #                       + tf.nn.l2_loss(real_feats[1] - fake_feats[1]) \
    #                       + tf.nn.l2_loss(real_feats[2] - fake_feats[2]) \
    #                       + tf.nn.l2_loss(real_feats[3] - fake_feats[3]) \
    #                       + tf.nn.l2_loss(real_feats[4] - fake_feats[4]))
    pair_loss = tf.reduce_mean(tf.square(tf.matmul(real_hash1, tf.transpose(real_hash1)) - similairy))+tf.reduce_mean(
        tf.square(tf.sign(real_hash1)-real_hash1)
    )

    MSE_loss1 = tf.nn.l2_loss(x64 - gen_img)   #/batch_size #/64/64/3
    MSE_loss2 = tf.reduce_mean(tf.abs(x64 - gen_img2))
    # hash_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=hash1, labels=real_hash2))
    # hash_loss = tf.nn.l2_loss(hash2 - tf.sign(hash2))
    D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=tf.ones_like(dis_real)))
    D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake,  labels=tf.zeros_like(dis_fake)))
    D_loss = D_fake+ D_real
    G_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake,  labels=tf.ones_like(dis_fake)))

    return MSE_loss1,MSE_loss2, D_loss, G_loss,pair_loss



batch_size = 50
graph = tf.Graph()
import sys
hidden_size = int(sys.argv[1])
with graph.as_default():
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    lr_D = tf.placeholder(tf.float32, shape=[])
    lr_S = tf.placeholder(tf.float32, shape=[])
    lr_E = tf.placeholder(tf.float32, shape=[])
    weight_H = tf.placeholder(tf.float32, shape=[])
    opt_D = tf.train.AdamOptimizer(lr_D, epsilon=1.0)
    opt_S = tf.train.AdamOptimizer(lr_S, epsilon=1.0)
    opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)


with graph.as_default():
    tower_grads_e = []
    tower_grads_g = []
    tower_grads_d = []
    all_input64 = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
    all_input224 = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    sim_metric = tf.placeholder(tf.float32, [batch_size, batch_size])
    train_model = tf.placeholder(tf.bool)

    with tf.device('/gpu:1'):
        with tf.name_scope('Tower_0') as scope:
            hiden_layer1,hiden_layer2, gen_img1, dis_fake1, dis_real1,super_img= inference(all_input64,all_input224)
            SSE_loss1,SSE_loss2, D_loss, G_loss ,N_loss= loss(all_input64, gen_img1,super_img,dis_fake1,dis_real1,hiden_layer1,hiden_layer2,sim_metric)
            params = tf.trainable_variables()
            E_params = [i for i in params if 'enc' in i.name]
            G_params = [i for i in params if 'gen' in i.name]
            D_params = [i for i in params if 'dis' in i.name]
            S_params = [i for i in params if 'srez' in i.name]
            reg_e = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5),G_params+E_params)
            reg_d = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), D_params)
            reg_s = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), S_params)

            grads_e = opt_E.compute_gradients(SSE_loss1*0+reg_e*0 + N_loss,var_list=E_params)
            grads_d = opt_D.compute_gradients(D_loss + reg_d,var_list=D_params)
            grads_s = opt_D.compute_gradients(SSE_loss2 + reg_s+G_loss*0.1, var_list=S_params)


with graph.as_default():
    train_D = opt_D.apply_gradients(grads_d, global_step=global_step)
    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
    train_S = opt_E.apply_gradients(grads_s, global_step=global_step)

with graph.as_default():
    saver = tf.train.Saver()

    sess = tf.InteractiveSession(graph=graph,config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    #sess = tf.InteractiveSession(graph=graph,config=config)
    # saver.restore(sess, "checkpoint/0005")
    sess.run(tf.global_variables_initializer())


dataset1 = sio.loadmat('cifar-10.mat')['train_data']  # data_set
test_data_ =sio.loadmat('cifar-10.mat')['test_data']

img64 = []
img224 =[]
test_data = []
for i in tqdm.tqdm(range( dataset1.shape[-1])):
    t = dataset1[:, :, :, i]
    img2 = scipy.misc.imresize(t, [64, 64]).astype(np.float32)
    img3 = scipy.misc.imresize(t, [224, 224]).astype(np.float32)
    img64.append(img2/255.0)
    img224.append(img3/255.0)
test224 = []
for i in xrange(test_data_.shape[-1]):
    t = test_data_[:,:,:,i]
    image1 = scipy.misc.imresize(t, [64, 64]).astype(np.float32)
    image2 = scipy.misc.imresize(t, [224, 224]).astype(np.float32)
    test_data.append(image1/255.)
    test224.append(image2/255.)

test224 = np.array(test224)
test_data = np.array(test_data)
img64 = np.array(img64)
img224 = np.array(img224)
num_examples = len(img64)
total_batch = int(np.floor(num_examples / batch_size ))
epoch = 0
cur_epoch = 0
num_epochs = 5
e_learning_rate = 1e-3
globa_beta_indx = 0
start = 1
S =  np.load('S_gt.npz')['S']
while epoch < num_epochs:
    iter_ = data_iterator()
    for i in range(total_batch):
        cur_epoch += 1.0
        next_batches64 ,indx3= iter_.next()
        next_batches224 = img224[indx3]
        ss = S[indx3,:][:,indx3]
        # _,_,_,\
        # mse_loss1,mse_loss2,gloss,dloss,reconstructs,super_image,hd,neigh_loss= sess.run(
        #     [
        #         train_E,train_D,train_S,
        #         SSE_loss1,SSE_loss2,G_loss,D_loss,gen_img1,super_img,hiden_layer2,N_loss
        #     ],
        #     {
        #         lr_D :e_learning_rate,
        #         lr_S: e_learning_rate,
        #         lr_E: e_learning_rate,
        #         all_input64: next_batches64,
        #         all_input224: next_batches224,
        #         weight_H : start*epoch,
        #         sim_metric: ss,
        #         train_model: True
        #     }
        # )

        _, neigh_loss,hd = sess.run(
            [
                train_E,  # train_D,train_S,
                 N_loss,
                hiden_layer2
            ],
            {
                lr_D: e_learning_rate,
                lr_S: e_learning_rate,
                lr_E: e_learning_rate,
                all_input64: next_batches64,
                all_input224: next_batches224,
                train_model: True,
                sim_metric: ss

            }
        )


        print hd[1][:20]
        print "epoch:{0},D_loss:{1},G_loss:{2},MSE_loss:{3},MSE_super:{4}".format(cur_epoch / total_batch, dloss,gloss,mse_loss1 ,mse_loss2)
        # print "epoch:{0},N_loss:{1}".format(cur_epoch / total_batch, neigh_loss)
    epoch += 1
    # IPython.display.clear_output()
    # plot_network_output('figs/'+str(epoch).zfill(4)+'_test.jpg')
    # plot_network_output_train('figs/'+str(epoch).zfill(4)+'_train.jpg',next_batches64,reconstructs,super_image,mse_loss1,mse_loss2)
# saver.save(sess, 'checkpoint/'+str(epoch).zfill(4))

saveB('cifar_Neigh_'+str(epoch).zfill(4)+'_'+str(hidden_size).zfill(4))
