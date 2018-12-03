import pandas as pd 
import numpy as np
from PIL import Image
import math
import build_data
import os, sys
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

"""
    Train size:  42556 - 1000 = 41556
    Val size:  1000
    Total size:  192556 including noship (empty) images
"""
_NUM_SHARDS = 50
_SEED = 97
VAL_SIZE = 1000
FILTER_NOSHIP = True


def _convert_dataset(dataset_split):

    masks_rle = pd.read_csv(FLAGS.csv_file)

    filenames = os.listdir(FLAGS.image_folder)
    # random shuffle images 
    random.shuffle(filenames)

    # perform train-val split based the same constant seed number
    train_imgs, val_imgs = train_test_split(filenames, test_size=VAL_SIZE, random_state=_SEED)
    if dataset_split == 'train':
        filenames = train_imgs
    elif dataset_split == 'val':
        filenames = val_imgs
    else:
        raise TypeError('Unknown dataset split')
    tf.logging.info('dataset split: {}   dataset size: {}'.format(dataset_split, len(filenames)))
    tf.logging.info('dataset files are like: {}'.format(filenames[:5]))
    num_images = len(filenames)

    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)
    num_images_converted = 0

    # https://www.kaggle.com/inversion/run-length-decoding-quick-start
    # see visualization: https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization
    def rle_decode(mask_rle, shape=(768, 768)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background

        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T  # Needed to align to RLE direction

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                sys.stdout.flush()
                imageId = filenames[i]
                mask = masks_rle.loc[masks_rle['ImageId'] == imageId, 'EncodedPixels'].dropna().tolist()
                # filter emtpy images first
                if FILTER_NOSHIP and len(mask) == 0:
                    continue

                image_filename = os.path.join(FLAGS.image_folder, filenames[i])
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                img_height, img_width = image_reader.read_image_dims(image_data)

                # start parsing masks
                seg_data = np.zeros((768, 768))

                for m in mask:
                    seg_data += rle_decode(m)
                # save the shit to a temporary location
                Image.fromarray(seg_data, mode='L').save('/tmp/tmp.png')
                seg_data_png = tf.gfile.FastGFile('/tmp/tmp.png', 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data_png)
                height, width = 768, 768
                if seg_height != height or seg_width != width or img_height != height or img_width != width:
                    raise RuntimeError('bullshit shape!!!')
                example = build_data.image_seg_to_tfexample(image_data, filenames[i], height, width, seg_data_png)
                tfrecord_writer.write(example.SerializeToString())
                num_images_converted += 1
                #tf.logging.debug('finished image: {}'.format(imageId))
    sys.stdout.write('\n')
    sys.stdout.flush()
    tf.logging.info('actual number of images converted: {}'.format(num_images_converted))
    
def main(unused_argv): 
    if len(os.listdir(FLAGS.output_dir)) != 0:
        raise RuntimeError('Remember to clear output dir: {} before you run this'.format(FLAGS.output_dir))
    _convert_dataset('train')
    _convert_dataset('val')


if __name__ == '__main__':
  tf.app.run()
             
            
