import matplotlib
matplotlib.use('pdf')
import os
import tqdm
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.python.framework import function
from generator import Vgg19
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
def generator2(features, channels=3):

    # Upside-down all-convolutional resnet

    mapsize = 3
    res_units = [512, 256, 128,64,32]  #[256, 128, 96,64]

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
    layers = [64, 128, 256, 40]

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
    # model.add_batch_norm()
    # model.add_relu()
    # print model.get_output().shape
    # raw_input()
    # model.add_flatten()
    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    # model.add_dense(1024)
    # model.add_batch_norm()
    # model.add_relu()
    # model.add_dense(hidden_size)

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
            # hiden_layer__,_= encode(x64)
            vgg_net = Vgg19('./vgg19.npy', codelen=hidden_size)
            print xx224.get_shape()
            hiden_layer__ = vgg_net.build_com(xx224)
            feat = vgg_net.fc7
            hepsilon = tf.ones(shape=tf.shape(hiden_layer__), dtype=tf.float32) * .5
            yout, pout = DoublySN(hiden_layer__, hepsilon)
            yout = tf.reshape(yout,hiden_layer__.get_shape())
        with tf.variable_scope("gen"):
            gen_img,_ = generator2(hiden_layer__)

        return  gen_img,yout,feat


def loss(x64, gen_img):
    # feat_loss =  0.00001*(tf.nn.l2_loss(real_feats[0]-fake_feats[0]) \
    #                       + tf.nn.l2_loss(real_feats[1] - fake_feats[1]) \
    #                       + tf.nn.l2_loss(real_feats[2] - fake_feats[2]) \
    #                       + tf.nn.l2_loss(real_feats[3] - fake_feats[3]) \
    #                       + tf.nn.l2_loss(real_feats[4] - fake_feats[4]))


    MSE_loss1 = tf.reduce_sum(tf.abs(x64 - gen_img))   #/batch_size #/64/64/3
    Mean_sse = tf.reduce_mean(tf.abs(x64 - gen_img))
    return MSE_loss1,Mean_sse



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
            gen_img1,compress_code,feats= inference(all_input64,all_input224)
            SSE_loss1, msse= loss(all_input64, gen_img1)
            params = tf.trainable_variables()
            E_params = [i for i in params if 'enc' in i.name]
            G_params = [i for i in params if 'gen' in i.name]
            grads_e = opt_E.compute_gradients(SSE_loss1,var_list=E_params+G_params)

with graph.as_default():
    # train_D = opt_D.apply_gradients(grads_d, global_step=global_step)
    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
    # train_S = opt_E.apply_gradients(grads_s, global_step=global_step)

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
num_epochs = 500
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
        _,recons, mse_loss1,compress_code2,mean_ss,feats4= sess.run(
            [
                train_E,
                gen_img1,
                SSE_loss1,compress_code, msse,feats
            ],
            {

                lr_E: e_learning_rate,
                all_input64: next_batches64,
                all_input224: next_batches224,
                train_model: True
            }
        )
        print feats4.shape
        # _, neigh_loss,hd = sess.run(
        #     [
        #         train_E,  # train_D,train_S,
        #          N_loss,
        #         hiden_layer2
        #     ],
        #     {
        #         lr_D: e_learning_rate,
        #         lr_S: e_learning_rate,
        #         lr_E: e_learning_rate,
        #         all_input64: next_batches64,
        #         all_input224: next_batches224,
        #         train_model: True,
        #         sim_metric: ss
        #
        #     }
        # )



        print "epoch:{0},Sum_MSE_loss:{1}, Mean_sse: {2}".format(cur_epoch / total_batch,mse_loss1,mean_ss )


        # print "epoch:{0},N_loss:{1}".format(cur_epoch / total_batch, neigh_loss)
    epoch += 1
    # IPython.display.clear_output()
    # plot_network_output('figs/'+str(epoch).zfill(4)+'_test.jpg')
    plot_network_output_train('figs/'+str(epoch).zfill(4)+'_train.jpg',next_batches64,recons,recons,mse_loss1,mse_loss1)
# saver.save(sess, 'checkpoint/'+str(epoch).zfill(4))

# saveB('cifar_Neigh_'+str(epoch).zfill(4)+'_'+str(hidden_size).zfill(4))
