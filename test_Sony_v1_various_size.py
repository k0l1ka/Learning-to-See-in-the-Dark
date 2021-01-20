from __future__ import division
import os, sys
import tensorflow as tf
import numpy as np
import rawpy
import glob
import PIL
tf.compat.v1.disable_eager_execution()

input_dir = './../../dataset/Sony_full/short_full/'
gt_dir = './../../dataset/Sony_full/long_full/'

if len(sys.argv) == 1:
    sys.exit("Pass the number of scenes in train set as an argument for puthon script!")
# print(sys.argv[1])
size = int(sys.argv[1])

checkpoint_dir = './v1_checkpoint_Sony_on_train_' + str(size) + '/'
result_dir = './output_after_train_' + str(size) + '/' 


# get test IDs
test_fns = glob.glob(gt_dir + '1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

print("test set of " + str(len(test_ids)) + "\n")

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = tf.compat.v1.layers.conv2d(input, 32, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv1 = tf.compat.v1.layers.conv2d(conv1, 32, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    pool1 = tf.compat.v1.layers.max_pooling2d(conv1, 2, 2, padding='same')

    conv2 = tf.compat.v1.layers.conv2d(pool1, 64, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv2 = tf.compat.v1.layers.conv2d(conv2, 64, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    pool2 = tf.compat.v1.layers.max_pooling2d(conv2, 2, 2, padding='same')

    conv3 = tf.compat.v1.layers.conv2d(pool2, 128, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv3 = tf.compat.v1.layers.conv2d(conv3, 128, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    pool3 = tf.compat.v1.layers.max_pooling2d(conv3, 2, 2, padding='same')

    conv4 = tf.compat.v1.layers.conv2d(pool3, 256, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv4 = tf.compat.v1.layers.conv2d(conv4, 256, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    pool4 = tf.compat.v1.layers.max_pooling2d(conv4, 2, 2, padding='same')

    conv5 = tf.compat.v1.layers.conv2d(pool4, 512, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv5 = tf.compat.v1.layers.conv2d(conv5, 512, [3, 3], padding='same', dilation_rate=1, activation=lrelu)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = tf.compat.v1.layers.conv2d(up6, 256, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv6 = tf.compat.v1.layers.conv2d(conv6, 256, [3, 3], padding='same', dilation_rate=1, activation=lrelu)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = tf.compat.v1.layers.conv2d(up7, 128, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv7 = tf.compat.v1.layers.conv2d(conv7, 128, [3, 3], padding='same', dilation_rate=1, activation=lrelu)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = tf.compat.v1.layers.conv2d(up8, 64, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv8 = tf.compat.v1.layers.conv2d(conv8, 64, [3, 3], padding='same', dilation_rate=1, activation=lrelu)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = tf.compat.v1.layers.conv2d(up9, 32, [3, 3], padding='same', dilation_rate=1, activation=lrelu)
    conv9 = tf.compat.v1.layers.conv2d(conv9, 32, [3, 3], padding='same', dilation_rate=1, activation=lrelu)

    conv10 = tf.compat.v1.layers.conv2d(conv9, 12, [1, 1], padding='same', dilation_rate=1, activation=lrelu)

    out = tf.compat.v1.depth_to_space(input=conv10, block_size=2)
    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


sess = tf.compat.v1.Session()
in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

for test_id in test_ids:
    # print("test_id= " + str(test_id) + "\n")
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        # print("k= " + str(k) + "\n")
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the groundtruth

        PIL.Image.fromarray((output * 255).astype(np.uint8)).save(result_dir + '%5d_00_%d_out.png' % (test_id, ratio))
        # PIL.Image.fromarray((scale_full * 255).astype(np.uint8)).save(result_dir + '%5d_00_%d_scale.png' % (test_id, ratio))
        PIL.Image.fromarray((gt_full * 255).astype(np.uint8)).save(result_dir + '%5d_00_%d_gt.png' % (test_id, ratio))
