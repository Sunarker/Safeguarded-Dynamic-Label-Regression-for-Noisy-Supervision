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

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'events_bootstrapping/cifar10_train',
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

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      #indices, images, labels = cifar10.distorted_inputs()
      indices, images, labels, T, T_mask = cifar10.noisy_distorted_inputs(noise_ratio=FLAGS.noise_ratio,return_T_flag=True)
 
    # Build a Graph that computes the logits predictions from the
    # inference model.
    is_training = tf.placeholder(tf.bool,shape=(),name="bn_flag")
    logits = cifar10.inference(images,training=is_training)

    # Calculate loss.
    # loss = cifar10.loss(logits, labels)
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.int64), logits=logits, name='cross_entropy_per_example')
    loss1 = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', loss1)

    # perceptual loss
    preds = tf.nn.softmax(logits)
    preds = tf.clip_by_value(preds, 1e-8, 1-1e-8)
    loss2 = tf.reduce_mean(-tf.reduce_sum(preds*tf.log(preds),axis=-1), name='perceptual_certainty_soft')
    #loss2 = tf.reduce_mean(-tf.reduce_sum(tf.stop_gradient(tf.to_float(tf.one_hot(tf.argmax(preds,axis=-1),depth=cifar10.NUM_CLASSES,axis=-1)))*tf.log(preds),axis=-1), name='perceptual_certainty_hard')
    tf.add_to_collection('losses', loss2)

    # l2 loss
    l2_loss = tf.add_n([cifar10.WEIGHT_DECAY * tf.nn.l2_loss(tf.cast(v, tf.float32))
                             for v in tf.trainable_variables() if 'batch_normalization' not in v.name],name='l2_loss')
    tf.add_to_collection('losses', l2_loss)

    # weighted loss
    alpha = tf.placeholder(tf.float32, shape=(), name='perceptual_weight')
    _LoggerHook_loss = alpha * loss1 + (1-alpha) * loss2
    loss = alpha * loss1 + (1-alpha) * loss2 + l2_loss

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Calculate prediction
    # acc_op contains acc and update_op. So it is the cumulative accuracy when sess runs acc_op
    # if you only want to inspect acc of each batch, just sess run acc_op[0]
    acc_op = tf.metrics.accuracy(labels, tf.argmax(logits,axis=1))
    tf.summary.scalar('training accuracy', acc_op[0])

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
                               
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=60,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:
      while not mon_sess.should_stop():
        #mon_sess.run([train_op,acc_op,global_step],feed_dict={is_training:True, alpha: 0.5})      
        res = mon_sess.run([train_op,acc_op,global_step,T,T_mask],feed_dict={is_training:True, alpha:0.5})
        if res[2] % 1000 == 0:
           print('Disturbing matrix\n',res[3])
           print('Masked structure\n',res[4])


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
