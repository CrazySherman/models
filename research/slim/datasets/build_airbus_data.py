import pandas as pd 
import numpy as np
import math
import os, sys
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

import dataset_utils


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir',
                           './airbus/tfrecords',
                           'outupt tfrecords directory')

tf.app.flags.DEFINE_string('image_folder',
                           None,
                           'Folder containing images.')

tf.app.flags.DEFINE_string('csv_file',
                            None,
                            'segmentation csv file')

tf.app.flags.DEFINE_integer('num_shards',
                            50,
                            'number of shards for train/val set')

"""
    Train size:  42556 - 1000 = 41556
    Val size:  1000
    Total size:  192556 including noship (empty) images
"""
_SEED = 97
_IMAGE_SIZE = 768
_NUM_CHANNELS = 3
# The names of the classes.
_CLASS_NAMES = [
    'noship',
    'ship'
]

masks_rle = pd.read_csv(FLAGS.csv_file)

def _extract_image(filename):
    """Extract the images into a numpy array.

    Args:
    filename: The path to an airbus images file.

    Returns:
    A binary format of image jpg string
    """
    # print('Extracting images from: ', filename)
    return tf.gfile.FastGFile(image_filename, 'rb').read()


def _extract_label(imageId):
    """Extract label for the given image id

    Returns:
    A numpy array of shape [number_of_labels]
    """
    # print('Extracting labels for image: ', imageId)
    mask = masks_rle.loc[masks_rle['ImageId'] == imageId, 'EncodedPixels'].dropna().tolist()
    if len(mask) == 0:
        return 0
    else:
        return 1

def _prepare_filenames(image_folder, dataset_split, seed):
    filenames = os.listdir(image_folder)

    # perform train-val split based the same constant seed number
    train_imgs, val_imgs = train_test_split(filenames, test_size=0.1, random_state=seed)
    if dataset_split == 'train':
        filenames = train_imgs
    elif dataset_split == 'val':
        filenames = val_imgs
    else:
        raise TypeError('Unknown dataset split')
    tf.logging.info('dataset split: {}   dataset size: {}'.format(dataset_split, len(filenames)))
    tf.logging.info('dataset files are like: {}'.format(filenames[:5]))
    return filenames

def _convert_dataset(dataset_split):

    filenames = _prepare_filenames(FLAGS.image_folder, dataset_split, _SEED)
    num_images = len(filenames)
    num_shards = FLAGS.num_shards
    num_per_shard = int(math.ceil(num_images / float(num_shards)))
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=(_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS))
        encoded_png = tf.image.encode_png(image)
    
        with tf.Session('') as sess:
            for shard_id in range(num_shards):
                output_filename = os.path.join(FLAGS.output_dir,'%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, num_shards))
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_idx = shard_id * num_per_shard
                    end_idx = min((shard_id + 1) * num_per_shard, num_images)
                    for i in range(start_idx, end_idx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()
                        image_filename = os.path.join(FLAGS.image_folder, filenames[i])
                        image_id = filenames[i]
                        img_data = _extract_image(image_filename)
                        label = _extract_label(image_id)
                        # png_string = sess.run(encoded_png, feed_dict={image: img})
                        example = dataset_utils.image_to_tfexample(img_data, 'jpeg'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, label)
                        tfrecord_writer.write(example.SerializeToString())

    print('Finished processing set: ', dataset_split, ' coverted images: ', num_images)



    
def main(unused_argv): 
    if len(os.listdir(FLAGS.output_dir)) != 0:
        raise RuntimeError('Remember to clear output dir: {} before you run this'.format(FLAGS.output_dir))
    _convert_dataset('train')
    _convert_dataset('val')
    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, FLAGS.output_dir)
    print('finished converting airbus dataset')


if __name__ == '__main__':
  tf.app.run()
             
            
