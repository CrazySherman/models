# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for training."""

import six
import sys
import tensorflow as tf
from deeplab.core import preprocess_utils

slim = tf.contrib.slim


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  scope=None):
  """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    loss_weights = tf.to_float(tf.not_equal(scaled_labels,
                                               ignore_label)) * loss_weight
    one_hot_labels = slim.one_hot_encoding(
        scaled_labels, num_classes, on_value=1.0, off_value=0.0)
    tf.losses.softmax_cross_entropy(
        one_hot_labels,
        tf.reshape(logits, shape=[-1, num_classes]),
        weights=loss_weights,
        scope=loss_scope)

# smoothed dice coefficient
def dice_coefficient(logits, labels, scope_name, padding_val=255):
    """
        logits: [batch_size * img_height * img_width * num_classes]
        labels: [batch_size * img_height * img_width]
    """
    with tf.name_scope(scope_name, 'dice_coef', [logits, labels]) as scope:
        sm = tf.nn.softmax(logits)
        preds = sm[:,:,:,1]
        # remove padded parts 
        padded = tf.cast(tf.not_equal(labels, padding_val), tf.int32)
        preds = tf.to_float(padded) * preds
        labels = padded * labels
        probe = tf.argmax(sm, axis=-1)
        probe1 = tfcount(probe, 1, 'count1')
        probe2 = tfcount(probe, 0, 'count0')
        addc(probe1)
        addc(probe2)
        # flat thee shit
        batch_size = logits.shape[0]
        preds = tf.reshape(preds, shape=[batch_size, -1])
        labels = tf.reshape(tf.to_float(labels), shape=[batch_size, -1])
        dices = (1 + 2 * tf.reduce_sum(preds * labels, axis=1)) / (1 + tf.reduce_sum(preds, axis=1) + tf.reduce_sum(labels, axis=1))
        return tf.reduce_mean(dices)

# def bbox_classifier(fea_map, labels, scope_name):
#   """Args:
#       fea_map: [N * height * width * num_classes]
#       labels: [N]
#     Return:
#       classifier logits [N * 2]
#   """
#     tf.nn.Conv2d(3)
#     tf.nn.Conv2d(3)
#     tf.nn.relu()
#     tf.nn.Conv2d(1)
#     return tf.reduce_sum()


# # bbox loss
# def bbox_loss(logits, labels, scope_name, padding_val=None):
#     """Args
#         logits: [batch_size * img_height * img_width * num_classes]
#         labels: [batch_size * img_height * img_width]
#       Return: loss
#     """
#     with tf.name_scope(scope_name, 'bbox_classifier', [fea_map, labels]) as scope:
#       fea_map, labels = nms(logits, labels)
#       classifier_logits = bbox_classifier(fea_map, logits)
#       return tf.nn.add_softmax_cross_entropy_loss(classifier_logits)

def negative_sampling(labels, logits):
    """Args:
      negatively sample imbalanced masks,
      inputs: labels and logits are flattened with shape [N, num_class], logits already softmaxed
    Return:
      sampled labels and logits tensor
    """
    # positive samples
    pos_indices = tf.greater(logits[:,1], logits[:,0])

    # negative samples
    neg_indices = tf.greater(logits[:,0], logits[:,1])
    # all_indices = tf.cat([pos_indices, tf.random(neg_indices)])
    all_indices = pos_indices  # this is targeting false positive only

    return tf.boolean_mask(labels, all_indices), tf.boolean_mask(logits, all_indices)

def focal_loss(labels, logits, scope_name=None, gamma=5, alpha=10, padding_val=255):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: flattened one hot label [N, num_class]
    :param logits: flattened logits before softmax layer of shape [N, num_class]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    #with name_scope(scope_name, 'focal_loss', )
    epsilon = 1.e-9
    logits = tf.nn.softmax(logits)    
    logits, labels = negative_sampling(labels, logits)
    addc(logits.shape)
    model_out = logits + epsilon
    ce = -labels * tf.log(model_out)
    weight = tf.pow(1 - model_out, gamma)
    fl = alpha * weight * ce
    # remove padding vals
    notpadded = tf.cast(tf.not_equal(labels, padding_val), tf.float32)
    fl = fl * notpadded
    reduced_fl = tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return reduced_fl

def tfcount(labels, val, name='count'):
    return tf.reduce_sum(tf.cast(tf.equal(labels, val), tf.int32), name=name)

def addc(probe):
    tf.add_to_collection('debugging', probe)

# experiment different types of losses
def my_mixed_loss(scales_to_logits,
                  labels,
                  num_classes,
                  ignore_label,
                  loss_weight=1.0,
                  upsample_logits=True,
                  scope=None):  
  """ 
    same interface as add_softmax.......
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')
  # TODO remove this probe ondce code cmoplete
  #probe1 = tfcount(labels, 1, 'count1')
  #probe2 = tfcount(labels, 255, 'count255')
  #addc(probe1)  
  #addc(probe2)

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    # caculate dice coefficient
    dice = dice_coefficient(logits, tf.squeeze(scaled_labels), loss_scope)
    dice_loss = tf.identity(-tf.log(dice), 'dice_coef')

    flattened_labels = tf.reshape(scaled_labels, shape=[-1])
    # weighted loss coefficient? 
    # loss_weight = tf.to_float(scaled_labels + 1)    # SET pos:neg loss contrib to 2: 1
    loss_weights = tf.to_float(tf.not_equal(flattened_labels,ignore_label)) * loss_weight

    one_hot_labels = slim.one_hot_encoding(
        flattened_labels, num_classes, on_value=1.0, off_value=0.0)
    logits_flattened = tf.reshape(logits, shape=[-1, num_classes])

    # bce loss
    bce_loss = 1e-3 * tf.losses.softmax_cross_entropy(
        one_hot_labels,
        logits_flattened,
        weights=loss_weights,
        scope=loss_scope,
        loss_collection=None)    # do not let this shit add otherwise it'll be collected by trainer
    # # focal loss
    f_losses = focal_loss(one_hot_labels, logits_flattened)
    f_losses = tf.identity(f_losses, 'focal_loss')


    # bce ~ 0.25 ~ 1.5,   dice ~ 0.8

    # the funny shit is the deployment module will aggregate all losses you added by a simple sum
    # so you gotta make sure you have all loss individually defined well here
    #slim.losses.add_loss(bce_loss)
    slim.losses.add_loss(f_losses)
    slim.losses.add_loss(dice_loss)   # log the dice to smooth its gradient: https://www.kaggle.com/iafoss/unet34-dice-0-87/notebook

def my_miou(predictions, labels):
  """Calculates mean iou ops, the tf.metrics.mean_iou is confusing and had issues when using it
  Args:
    predictions: [batch, num_bits], 0/1 masks
    labels: [batch, num_bits], 0/1 masks.
  
  Returns:
    ops that returns a mean_iou value for a single class, in this case, the ship class
    we do not calculates miou for background class cuz that's obviously close to 1 
  """
  assert_op_pred = tf.Assert(tf.less_equal(tf.reduce_max(predictions), 1), [predictions])
  assert_op_label = tf.Assert(tf.less_equal(tf.reduce_max(labels), 1), [labels])

  with tf.control_dependencies([assert_op_label, assert_op_pred]):
    nom = tf.reduce_mean(1 + 2 * tf.reduce_sum(predictions * labels, axis=-1)) 
    denom = 1 + tf.reduce_sum(predictions, axis=-1) + tf.reduce_sum(labels, axis=-1)
    return nom/denom



def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)

  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

  if variables_to_restore:
    return slim.assign_from_checkpoint_fn(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
  return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in slim.get_model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(
    learning_policy, base_learning_rate, learning_rate_decay_step,
    learning_rate_decay_factor, training_number_of_steps, learning_power,
    slow_start_step, slow_start_learning_rate):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                  learning_rate)
