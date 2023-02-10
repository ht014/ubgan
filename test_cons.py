import matplotlib
matplotlib.use('pdf')
from BGAN.generator import Vgg19
import os
import tqdm
import scipy
import scipy.io as sio
import prettytensor as pt
import matplotlib.pyplot as plt
import IPython.display

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def data_iterator():
    while True:
        idxs = np.arange(0, len(img224))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(img224), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = img224[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs

from BGAN.srez_model import *
def generator2( features, channels=3):
        mapsize = 2
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

def encode(disc_input):
    mapsize = 2
    layers = [64, 128, 256, 512]

    old_vars = tf.all_variables()

    model = Model('DIS',  disc_input )

    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0

        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()
    model.add_flatten()
    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    model.add_dense(2048)
    model.add_batch_norm()
    model.add_relu()
    model.add_dense(hidden_size)

    new_vars = tf.all_variables()
    disc_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), disc_vars

def discriminator2(disc_input):
    # Fully convolutional model
    mapsize = 2
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

def plot_network_output(name_fig,real_image,re_image):

    next_batches64 = real_image
    recon_x=re_image
    examples = 13
    recon_x = np.squeeze(recon_x)

    fig, ax = plt.subplots(nrows=2,ncols=examples, figsize=(18,6))
    for i in xrange(examples):

        ax[(0,i)].imshow(create_image(recon_x[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(1,i)].imshow(create_image(next_batches64[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,i)].axis('off')
        ax[(1,i)].axis('off')
        #ax[(2,i)].axis('off')
    fig.suptitle('Bottom: inputs | Top: reconstructions')
    plt.savefig(name_fig)
    plt.close()


def inference(x64):
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                           batch_normalize=False,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with tf.variable_scope("enc"):
                # vgg_net = Vgg19('./vgg19.npy', codelen=hidden_size)
                # vgg_net.build(x224, beta_nima, train_model)
                hiden_layer,_= encode(x64)

        with tf.variable_scope("gen"):
            wdecode = tf.Variable(tf.random_normal([hidden_size, 8*8*256], stddev=1.0 / tf.sqrt(float(4096)), dtype=tf.float32),name='wdecode')
            baise = tf.Variable(tf.random_normal([8*8*256], stddev=1.0 / tf.sqrt(float(4093)), dtype=tf.float32),name='bias')

            new_space_ = tf.matmul(hiden_layer, wdecode) + baise

            new_space = tf.nn.relu(tf.reshape(new_space_,[batch_size,8,8,256]))
            gen_img,_ = generator2(new_space)

            # x_tilde = generator(z_x_mean)
        with tf.variable_scope("dis"):
            dis_fake,_ = discriminator2(gen_img)
        # with tf.variable_scope("gen", reuse=True):
        #     x_p = generator(z_p)
        with tf.variable_scope("dis", reuse=True):
            dis_real, _ = discriminator2(x64)
        # with tf.variable_scope("dis", reuse=True):
        #     d_x_p, _ = discriminator(x_p)
        with tf.variable_scope('feat'):
            vgg_net = Vgg19('./vgg19.npy', codelen=hidden_size)
            real_feats = vgg_net.build(x64)
        with tf.variable_scope('feat',reuse=True):
            fake_feats = vgg_net.build(gen_img)
        return hiden_layer,gen_img,dis_fake,dis_real,real_feats,fake_feats


def loss(x64, gen_img,dis_fake,dis_real,real_feats,fake_feats):
    feat_loss =  0.00001*(tf.nn.l2_loss(real_feats[0]-fake_feats[0])\
    + tf.nn.l2_loss(real_feats[1] - fake_feats[1])\
    + tf.nn.l2_loss(real_feats[2] - fake_feats[2])\
    + tf.nn.l2_loss(real_feats[3] - fake_feats[3])\
    + tf.nn.l2_loss(real_feats[4] - fake_feats[4]))
    MSE_loss = tf.reduce_sum(tf.square(x64 - gen_img))

    D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=tf.ones_like(dis_real)))
    D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake,  labels=tf.zeros_like(dis_fake)))
    D_loss = D_fake+ D_real

    G_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake,  labels=tf.ones_like(dis_fake)))

    return MSE_loss, D_loss, G_loss,feat_loss


batch_size = 50
graph = tf.Graph()
hidden_size = 48
mode_path = 'checkpoint/0101'
with graph.as_default():
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    lr_D = tf.placeholder(tf.float32, shape=[])
    lr_G = tf.placeholder(tf.float32, shape=[])
    lr_E = tf.placeholder(tf.float32, shape=[])
    opt_D = tf.train.AdamOptimizer(lr_D, epsilon=1.0)
    opt_G = tf.train.AdamOptimizer(lr_G, epsilon=1.0)
    opt_E = tf.train.AdamOptimizer(lr_G, epsilon=1.0)


with graph.as_default():

    all_input64 = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
    train_model = tf.placeholder(tf.bool)
    with tf.device('/gpu:0'):
        with tf.name_scope('Tower_0') as scope:
            hiden_layer1, gen_img1, dis_fake1, dis_real1,real_feats1,fake_feats1= inference(all_input64)
            SSE_loss, D_loss, G_loss, Vgg_loss = loss(all_input64, gen_img1, dis_fake1, dis_real1, real_feats1, fake_feats1)

with graph.as_default():
    saver = tf.train.Saver()
    sess = tf.InteractiveSession(graph=graph,config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    restore_saver = tf.train.import_meta_graph('checkpoint/0101.meta')
    saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))


dataset1 = sio.loadmat('cifar-10.mat')['train_data']
img224 = []
img64 = []
for i in tqdm.tqdm(range(1000)):
    t = dataset1[:, :, :, i]
    image1 = scipy.misc.imresize(t, [224, 224])
    img224.append(image1)
    img2 = scipy.misc.imresize(t, [64, 64]).astype(np.float32)
    img64.append(img2/255.0)

img224 = np.array(img224)
img64 = np.array(img64)
num_examples = len(img64)
total_batch = int(np.floor(num_examples / batch_size ))

num_test_file = 10
iter_ = data_iterator()
for i in range(total_batch):
    next_batches224 ,indx3= iter_.next()
    next_batches64 = img64[indx3]
    reconstructs,hd,MSE_loss=  sess.run(
        [
           gen_img1,hiden_layer1,SSE_loss
         ],
        {
            all_input64: next_batches64,
            train_model: True
        }
        )

    IPython.display.clear_output()
    plot_network_output('figs_test/'+str(i).zfill(4)+'.jpg', next_batches64, reconstructs)
    print 'figs_test/'+str(i).zfill(4)+'.jpg','saves done',MSE_loss
    if i > num_test_file:
        print 'Test done !'
        break
