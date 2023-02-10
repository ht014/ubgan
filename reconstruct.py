import matplotlib
matplotlib.use('pdf')
import os
import tqdm
import scipy
import scipy.io as sio

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
                           feed_dict= {all_input224: all})
        if i == 0:
            B = feature[0]
        else:
            B = np.concatenate((B, feature[0]), axis=0)

    for i in xrange(0, len(test_images), batch_size):
        all = test_images[i:i + batch_size]
        feature =  sess.run([hiden_layer2], \
                            feed_dict={all_input224: all })
        if i == 0:
            B_ = feature[0]
        else:
            B_ = np.concatenate((B_, feature[0]), axis=0)
    np.savez(name_B+'.npz', dataset=B, test=B_)
    print 'save done!'


def data_iterator():
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


from BGAN.srez_model import *
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






def inference(hash_codes):
        with tf.variable_scope("gen"):
            wdecode = tf.Variable(tf.random_normal([hidden_size, 8*8*64], stddev=1.0 / tf.sqrt(float(4096)), dtype=tf.float32),name='wdecode')
            baise = tf.Variable(tf.random_normal([8*8*64], stddev=1.0 / tf.sqrt(float(4096)), dtype=tf.float32),name='bias')
            new_space_ = tf.matmul(hash_codes, wdecode) + baise
            new_space =tf.reshape(tf.nn.relu(new_space_),[batch_size,8,8,64])
            gen_img,_ = generator2(new_space)
        with tf.variable_scope('srez'):
            gen_img2, _ = super_resolution(gen_img)

        return  gen_img,gen_img2



batch_size = 1
graph = tf.Graph()
import sys
hidden_size = int(sys.argv[1])



with graph.as_default():
    inpput_hash_codes = tf.placeholder(tf.float32, [1,hidden_size])

    with tf.device('/gpu:0'):
        with tf.name_scope('Tower_0') as scope:
            reconstructed_image,super_image = inference(inpput_hash_codes)


with graph.as_default():
    saver = tf.train.Saver()

    sess = tf.InteractiveSession(graph=graph,config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    #sess = tf.InteractiveSession(graph=graph,config=config)
    saver.restore(sess, "checkpoint/0150")
    # sess.run(tf.global_variables_initializer())
dataset1 = sio.loadmat('cifar-10.mat')['data_set']  #cifar-10 data
pic_id=13319
scipy.misc.imsave("ok.jpg",dataset1[:,:,:,pic_id])

feat = np.load('cifar_Neigh_0150_0512.npz')
dataset =  feat['dataset']
hhh=dataset[pic_id]

re_image,s_image = sess.run([reconstructed_image,super_image],{inpput_hash_codes:[hhh]})

scipy.misc.imsave("1.jpg",re_image[0,:,:,:])
scipy.misc.imsave("2.jpg",s_image[0,:,:,:])


