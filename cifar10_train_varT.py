# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math
import pickle

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('init_dir','events_ce/cifar10_train',
                           """Directory where to load the intializing weights""")
tf.app.flags.DEFINE_string('train_dir', 'events_varT/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")                            
tf.app.flags.DEFINE_float('noise_ratio', 0.3,
                            """noise ratio to be used.""")                            
import cifar10
max_steps = int(math.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*cifar10.NUM_EPOCHS/FLAGS.batch_size))
tf.app.flags.DEFINE_integer('max_steps', max_steps,
                            """Number of batches to run.""")
   
def train(T_fixed, T_init):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
   
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      #indices, images, labels = cifar10.distorted_inputs()
      indices, images, labels, T_tru,T_mask_tru = cifar10.noisy_distorted_inputs(return_T_flag=True,noise_ratio=FLAGS.noise_ratio)
      indices = indices[:,0]

    # Build a Graph that computes the logits predictions from the
    # inference model.
    is_training = tf.placeholder(tf.bool,shape=(),name='bn_flag')
    logits = cifar10.inference(images,training=is_training)
    preds = tf.nn.softmax(logits)

    # fixed adaption layer
    fixed_adaption_layer = tf.cast(tf.constant(T_fixed),tf.float32)

    # adaption layer
    logits_T = tf.get_variable('logits_T',shape=[cifar10.NUM_CLASSES,cifar10.NUM_CLASSES],initializer=tf.constant_initializer(np.log(T_init + 1e-8)))
    adaption_layer = tf.nn.softmax(logits_T)

    # label adaption
    is_use = tf.placeholder(tf.bool,shape=(),name='warming_up_flag') 
    adaption = tf.cond(is_use, lambda: fixed_adaption_layer, lambda: adaption_layer)
    preds_aug = tf.clip_by_value(tf.matmul(preds,adaption), 1e-8, 1.0 - 1e-8)
    logits_aug = tf.log(preds_aug)

    # Calculate loss.
    loss = cifar10.loss(logits_aug, labels)   
 
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Calculate prediction
    # acc_op contains acc and update_op. So it is the cumulative accuracy when sess runs acc_op
    # if you only want to inspect acc of each batch, just sess run acc_op[0]
    acc_op = tf.metrics.accuracy(labels, tf.argmax(logits,axis=1))
    tf.summary.scalar('training accuracy', acc_op[0])

    #### build scalffold for MonitoredTrainingSession to restore the variables you wish
    variables_to_restore = []
    #variables_to_restore += [var for var in tf.trainable_variables() if ('dense' not in var.name and 'logits_T' not in var.name)]
    variables_to_restore += [var for var in tf.trainable_variables() if 'logits_T' not in var.name]
    variables_to_restore += [g for g in tf.global_variables() if 'moving_mean' in g.name or 'moving_variance' in g.name]
    for var in variables_to_restore:
      print(var.name)
    ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
    init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
         ckpt.model_checkpoint_path, variables_to_restore)
    def InitAssignFn(scaffold,sess):
       sess.run(init_assign_op, init_feed_dict)

    scaffold = tf.train.Scaffold(saver=tf.train.Saver(), init_fn=InitAssignFn)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(tf.get_collection('losses')[0])  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        scaffold = scaffold,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=60,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:
      warming_up_step = 32000
      step = 0
      varT_rec = []
      varT_trans_trace = []
      while not mon_sess.should_stop():
        if step < warming_up_step:
          res = mon_sess.run([train_op,acc_op,global_step,fixed_adaption_layer,T_tru,T_mask_tru],feed_dict={is_training:True,is_use:True})
        else:
          res = mon_sess.run([train_op,acc_op,global_step,adaption_layer,T_tru,T_mask_tru],feed_dict={is_training:True,is_use:False})
        step = res[2]

        if step % 5000 == 0:
           varT_rec.append(res[3])

        if step == warming_up_step:
           trans_before = res[3].copy()
        if step > warming_up_step:
           trans_after = res[3].copy()
           trans_gap = np.sum(np.absolute(trans_before - trans_after))
           varT_trans_trace.append([step, trans_gap])

    with open('varT_learnt_%.2f.pkl'%FLAGS.noise_ratio,'w') as w:
      pickle.dump(varT_rec, w)

    with open('varT_transvar_trace_%.2f.pkl'%FLAGS.noise_ratio,'w') as w:
      pickle.dump(varT_trans_trace, w)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  with open('T_%.2f.pkl'%FLAGS.noise_ratio) as f:
     data = pickle.load(f)

  T = data[0]
  T2 = data[2]
  T3 = np.eye(cifar10.NUM_CLASSES)

  print('estimated matrix (I): ', T)
  print('estimated matrix (II): ', T2)
  print('estimated matrix (III): ', T3)

  #train(T_fixed=T, T_init=T)
  train(T_fixed=T2, T_init=T2)
  #train(T_fixed=T3, T_init=T3)

if __name__ == '__main__':
  tf.app.run()
