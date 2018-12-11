from datasets import segmentation_dataset
import tensorflow as tf
from utils import input_generator
from train import _build_deeplab as model_fn
from deployment import model_deploy
from deeplab import common
# this shit for distributed training 
from tensorflow.python.training import supervisor

FLAGS = tf.app.flags.FLAGS
# this is a shitty but in jupyter, add this flag to avoid error out
tf.app.flags.DEFINE_string('f', '', 'kernel')

slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue
## this enables injecting debugging breakpoints
# tf.enable_eager_execution()

# airbus 
dataset = segmentation_dataset.get_dataset('airbus', 'train', dataset_dir='datasets/airbus/tfrecords')

# pascal voc
dataset2 = segmentation_dataset.get_dataset('pascal_voc_seg', 'trainval', dataset_dir='datasets/pascal_voc_seg/tfrecord/')

# the input_generator gives you dequeue ops, so you need to feed this ops to a queue object
# prefetch_queue = slim.prefetch_queue

samples1 = input_generator.get(
        dataset,
        [513, 513],
        8,
        dataset_split='train')
        
samples2 = input_generator.get(
        dataset2,
        [513, 513],
        8,
        dataset_split='trainval')
# test on get train samples
with tf.Session() as sess:
    ini_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(ini_op)
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess,coord=coord)
    cur_example_batch = sess.run(samples1)
#         print(cur_example_batch)
    cur_img_batch = cur_example_batch['image']
    cur_label_batch = cur_example_batch['label']
    print(cur_img_batch.shape, ' : ', cur_label_batch.shape)
    coord.request_stop()
    coord.join(thread)

import numpy as np
labels = cur_example_batch['label']

#print(np.histogram(labels.flat))
label = labels[0]

print(label.dtype)
print(label.max())
print(label.min())
f  =  open('shit.txt', 'w+')
flattened = label.flat
for i in flattened:
    f.write(str(i))
