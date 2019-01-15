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
import os
import time
import math
import pickle
import sys
import struct

import tensorflow as tf
import tensorflow.contrib.slim as slim

OneHotCategorical = tf.contrib.distributions.OneHotCategorical

import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('init_dir','events_ce/cifar10_train',
                           """Directory where to load the intializing weights""")
tf.app.flags.DEFINE_string('train_dir', 'events_varC/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('groudtruth', False,
                            """Whether to use to the transition matrix in sampling.""")
tf.app.flags.DEFINE_boolean('labeltrace', False,
                            """Whether to trace the label correction per epoch.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")                            
tf.app.flags.DEFINE_float('noise_ratio', 0.3,
                            """noise ratio to be used.""")                            

import cifar10
max_steps = int(math.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*cifar10.NUM_EPOCHS/FLAGS.batch_size))
max_steps_per_epoch = int(math.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size))
tf.app.flags.DEFINE_integer('max_steps', max_steps,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps_per_epoch', max_steps_per_epoch,
                            """Number of batches to run per epoch.""")

def init_C():
  with tf.Graph().as_default():
    # tf always return the final batch even it is smaller than the batch_size of samples
    indices, images, labels = cifar10.inputs(eval_data=False,noise_ratio=FLAGS.noise_ratio)
    indices = indices[:,0] # rank 2 --> rank 1, i.e., (batch_size,1) --> (batch_size,)
    is_training = tf.placeholder(tf.bool)
    logits = cifar10.inference(images,training=is_training)
    labels_ = tf.nn.softmax(logits)

    variables_to_restore = []
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint files found')
        return

      inds = []
      preds = []
      annotations = []
      with tf.Session() as sess:
          ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
          if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
          else:
                print('No checkpoint file found')
                return

          # start the queue runner
          coord = tf.train.Coordinator()
          try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
                num_iter = int(math.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size))
                step = 0
                while step < num_iter:
                        #print('step: ', step)
                        res = sess.run([indices,labels_,labels],feed_dict={is_training:True})
                        inds.append(res[0])
                        preds.append(res[1])
                        annotations.append(res[2])
                        step += 1

          except Exception as e:
                coord.request_stop(e)

          coord.request_stop()
          coord.join(threads, stop_grace_period_secs=10)

  inds = np.concatenate(inds,axis=0)
  preds = np.concatenate(preds,axis=0)
  annotations = np.concatenate(annotations,axis=0)

  filter_set = set()
  length = inds.shape[0]
  delete_list = []
  print("input length:", length)
  for i in xrange(length):
    if inds[i] in filter_set:
      delete_list.append(i)
    else:
      filter_set.add(inds[i]) 
  inds = np.delete(inds,delete_list,0)
  preds = np.delete(preds,delete_list,0)
  annotations = np.delete(annotations,delete_list,0)
  
  est_C = np.zeros((cifar10.NUM_CLASSES,cifar10.NUM_CLASSES))
  for i in xrange(annotations.shape[0]):
    label_ = np.argmax(preds[i])
    label = annotations[i]
    est_C[label_][label] += 1

  return inds, preds, annotations, est_C
 
def train(infer_z, noisy_y, C, img_label):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      #indices, images, labels = cifar10.distorted_inputs()
      indices, images, labels, T_tru,T_mask_tru = cifar10.noisy_distorted_inputs(return_T_flag=True,noise_ratio=FLAGS.noise_ratio)
      indices = indices[:,0] # rank 2 --> rank 1, i.e., (batch_size,1) --> (batch_size,)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    is_training = tf.placeholder(tf.bool,shape=(),name='bn_flag')
    logits = cifar10.inference(images,training=is_training)
    preds = tf.nn.softmax(logits)

    # approximate Gibbs sampling
    T = tf.placeholder(tf.float32,shape=[cifar10.NUM_CLASSES,cifar10.NUM_CLASSES],name='transition')
    if FLAGS.groudtruth:
      unnorm_probs = preds * tf.gather(tf.transpose(T_tru,[1,0]),labels)
    else:
      unnorm_probs = preds * tf.gather(tf.transpose(T,[1,0]),labels)
       
    probs = unnorm_probs / tf.reduce_sum(unnorm_probs,axis=1,keepdims=True)
    sampler = OneHotCategorical(probs=probs)
    labels_ = tf.stop_gradient(tf.argmax(sampler.sample(),axis=1))
 
    loss = cifar10.loss(logits, labels_)

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
    #variables_to_restore += [var for var in tf.trainable_variables() if 'dense' not in var.name] # if final layer is not included
    variables_to_restore += tf.trainable_variables() # if final layer is included
    variables_to_restore += [g for g in tf.global_variables() if 'moving_mean' in g.name or 'moving_variance' in g.name]
    for var in variables_to_restore:
      print(var.name)
    #variables_to_restore = []
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
      ## initialize some params
      alpha = 1.0 
      C_init = C.copy()
      trans_init = (C + alpha) / np.sum(C + alpha, axis=1, keepdims=True)

      ## running setting
      warming_up_step = 20000
      step = 0
      freq_trans = 200
     
      ### warming up transition
      with open('T_%.2f.pkl'%FLAGS.noise_ratio) as f:
        data = pickle.load(f)
      trans_warming = data[2]  # trans_init or np.eye(cifar10.NUM_CLASSES)

      ## record and run
      exemplars = []
      label_trace_exemplars = [] 
      infer_z_probs = dict()
      trans_before_after_trace = [] 
      while not mon_sess.should_stop():
        if step % freq_trans == 0: # update transition matrix in each n steps
          trans = (C + alpha) / np.sum(C + alpha, axis=1, keepdims=True)

        if step < warming_up_step:
          res = mon_sess.run([train_op,acc_op,global_step,indices,labels,labels_,probs],feed_dict={is_training:True, T: trans_warming})
        else:
          res = mon_sess.run([train_op,acc_op,global_step,indices,labels,labels_,probs],feed_dict={is_training:True, T: trans})

        #print(res[3].shape)
        trans_before = (C + alpha) / np.sum(C + alpha, axis=1, keepdims=True)
        C_before = C.copy()
        for i in xrange(res[3].shape[0]):
          ind = res[3][i]
          #print(noisy_y[ind],res[4][i])
          assert noisy_y[ind] == res[4][i] 
          C[infer_z[ind]][noisy_y[ind]] -= 1
          assert C[infer_z[ind]][noisy_y[ind]] >= 0
          infer_z[ind] = res[5][i]
          infer_z_probs[ind] = res[6][i]
          C[infer_z[ind]][noisy_y[ind]] += 1
          #print(res[4][i],res[5][i])

        trans_after = (C + alpha) / np.sum(C + alpha, axis=1, keepdims=True)
        C_after = C.copy()
        trans_gap = np.sum(np.absolute(trans_after - trans_before))
        rou = np.sum(C_after - C_before, axis=-1)/np.sum(C_before + alpha, axis=-1)
        rou_ = np.sum(np.absolute(C_after - C_before), axis=-1)/np.sum(C_before + alpha, axis=-1)
        trans_bound = np.sum((np.absolute(rou)+rou_)/(1+rou)) 
        trans_before_after_trace.append([step, trans_gap,trans_bound])
        #print(trans_gap, trans_bound)

        step = res[2]
        if step % 1000 == 0:
          print('Counting matrix\n', C)
          print('Counting matrix\n', C_init)
          print('Transition matrix\n', trans)
          print('Transition matrix\n', trans_init)

        if step % 5000 == 0:
          exemplars.append([infer_z.copy().keys(), infer_z.copy().values(), C.copy()])

        if step % FLAGS.max_steps_per_epoch == 0:
          r_n = 0
          all_n = 0
          for key in infer_z.keys():
            if infer_z[key] == img_label[key]:
              r_n += 1
            all_n += 1
          acc = r_n / all_n
          #print('accuracy: %.2f'%acc) 
          label_trace_exemplars.append([infer_z.copy(),infer_z_probs.copy(),acc]) 

      if not FLAGS.groudtruth:
        with open('varC_learnt_%.2f.pkl'%FLAGS.noise_ratio,'w') as w:
          pickle.dump(exemplars,w)
      else:
        with open('varC_learnt_%.2f_tru.pkl'%FLAGS.noise_ratio,'w') as w:
          pickle.dump(exemplars,w)

      if FLAGS.labeltrace:
        with open('varC_label_trace_%.2f.pkl'%FLAGS.noise_ratio,'w') as w:
          pickle.dump([label_trace_exemplars, img_label],w)        

      with open('varC_transvar_trace_%.2f.pkl'%FLAGS.noise_ratio,'w') as w:
        pickle.dump(trans_before_after_trace,w)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  #if os.path.exists('varC.pkl'):
  #  with open('varC.pkl') as f:
  #    inds, preds, annotations, C = pickle.load(f)
  #else:
  #    inds, preds, annotations, C = init_C()
  #    with open('varC.pkl','w') as w:
  #      pickle.dump([inds, preds, annotations, C],w)
  inds, preds, annotations, C = init_C()
  with open('varC_%.2f.pkl'%FLAGS.noise_ratio,'w') as w:
    pickle.dump([inds, preds, annotations, C],w)

  print('indices \n', inds, inds.shape)
  print('predictions \n', np.argmax(preds,axis=1),preds.shape[0])
  print('annotations \n', annotations,annotations.shape)
  print('estimated Counting Matrix \n', C)
  infer_z = dict()
  noisy_y = dict()
  for e in xrange(len(inds)):
    #print(inds[e])
    infer_z[inds[e]] = np.argmax(preds[e])
    noisy_y[inds[e]] = annotations[e]

  #for key, value in infer_z.items():
  #  print(key, value)

  img_label = dict()
  for i in xrange(1,6):
    path = 'data/cifar10/cifar-10-batches-bin/data_batch_%d_with_index.bin'%i
    with open(path,'rb') as f:
      data = f.read(3077)
      while data:
        ind = struct.unpack('I', data[:4])
        ind = ind[0]
        label = ord(data[4])
        img_label[ind] = label
        data = f.read(3077)

  train(infer_z, noisy_y, C, img_label)

if __name__ == '__main__':
  tf.app.run()
