# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
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

"""A toy model using Mesh TensorFlow on GPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import numpy
import tensorflow.compat.v1 as tf

from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow_estimator.python.estimator import estimator as estimator_lib

FLAGS = flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 64, 'Training batch size.')
tf.flags.DEFINE_integer('io_size', 16, 'Number of channels per feature.')
tf.flags.DEFINE_integer('hidden_size', 16, 'Size of each hidden layer.')
tf.flags.DEFINE_integer('num_hidden_layers', 1, 'Number of layers.')
tf.flags.DEFINE_string('master_dtype', 'bfloat16', 'dtype for master vars.')
tf.flags.DEFINE_string('slice_dtype', 'float32', 'dtype for slice vars.')
tf.flags.DEFINE_string('activation_dtype', 'float32', 'dtype for activations.')
tf.flags.DEFINE_string('optimizer', 'SGD', 'optimizer (SGD or Adafactor).')
tf.flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
tf.flags.DEFINE_string('mesh_shape', 'all:8', 'mesh shape')
tf.flags.DEFINE_string('layout', 'hidden_odd:all', 'layout rules')
tf.flags.DEFINE_integer('iterations', 100,
                        'Number of iterations per training loop.')
tf.flags.DEFINE_integer('step_with_nan', -1,
                        'If >= 0, a NaN tensor is added in forward pass.')
tf.flags.DEFINE_integer('train_steps', 10000, 'max steps')
tf.flags.DEFINE_integer('steps_per_checkpoint', 200, 'steps_per_checkpoint')
tf.flags.DEFINE_string(
    'model_dir',
    default='',
    help='The directory where the model will be stored.')

class ToyModelInput(object):
  """Wrapper class that acts as the input_fn to Estimator."""

  def __init__(self):
    self._num_examples = 10000  # 10k
    self._images = numpy.random.uniform(
        0, 1.0, [self._num_examples, FLAGS.io_size]).astype(numpy.float32)
    self._labels = self._images
    logging.info('init ToyModelInput()')

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.estimator.tpu.RunConfig` for details.
    batch_size = params['batch_size']
    logging.info('call ToyModelInput() with batch size {}'.format(batch_size))

    ds = Dataset.from_tensor_slices((self._images, self._labels)).repeat()

    dataset = ds.batch(batch_size, drop_remainder=True).prefetch(2)

    return dataset


def toy_model(features, mesh):
  """A toy model implemented by mesh tensorlfow."""
  batch_dim = mtf.Dimension('batch', FLAGS.batch_size)
  io_dim = mtf.Dimension('io', FLAGS.io_size)

  master_dtype = tf.as_dtype(FLAGS.master_dtype)
  slice_dtype = tf.as_dtype(FLAGS.slice_dtype)
  activation_dtype = tf.as_dtype(FLAGS.activation_dtype)

  x = mtf.import_tf_tensor(mesh, features, mtf.Shape([batch_dim, io_dim]))
  x = mtf.cast(x, activation_dtype)
  h = x
  for lnum in range(1, FLAGS.num_hidden_layers + 2):
    if lnum + 1 == FLAGS.num_hidden_layers + 2:
      # output layer
      dim = io_dim
    elif lnum % 2 == 0:
      dim = mtf.Dimension('hidden_even', FLAGS.hidden_size)
    else:
      dim = mtf.Dimension('hidden_odd', FLAGS.hidden_size)
    h = mtf.layers.dense(
        h, dim,
        use_bias=False,
        master_dtype=master_dtype,
        slice_dtype=slice_dtype,
        name='layer_%d' % lnum)
  y = h
  g = tf.train.get_global_step()
  if FLAGS.step_with_nan >= 0:
    # Trigger NaN in the forward pass, this is used for testing whether
    # MeshTensorFlow can handle occasional NaN value.
    y += mtf.import_tf_tensor(
        mesh,
        tf.divide(
            0.0,
            tf.cond(tf.equal(g, FLAGS.step_with_nan), lambda: 0., lambda: 1.)),
        mtf.Shape([]))

  loss = mtf.reduce_mean(mtf.square(y - x))
  return y, loss


def model_fn(features, labels, mode, params):
  """A model is called by Estimator."""
  del labels
  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)

  mesh_impl = HvdSimdMeshImpl(mtf.convert_to_shape(mesh_shape), 
                              mtf.convert_to_layout_rules(layout_rules))
  mesh = mtf.Mesh(graph, 'my_mesh')

  with mtf.utils.outside_all_rewrites():
    logits, loss = toy_model(features, mesh)

  # TRAIN mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients([loss],
                              [v.outputs[0] for v in graph.trainable_variables])
    if FLAGS.optimizer == 'Adafactor':
      optimizer = mtf.optimize.AdafactorOptimizer()
    else:
      assert FLAGS.optimizer == 'SGD'
      optimizer = mtf.optimize.SgdOptimizer(learning_rate=FLAGS.lr)
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
  else:
    # for now, we can only export fully-replicated tensors.
    fully_replicated_logits = mtf.anonymize(logits)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  tf_loss = tf.to_float(lowering.export_to_tf_tensor(loss))

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    tf.logging.info('tf_update_ops: {}'.format(tf_update_ops))
    train_op = tf.group(tf_update_ops)
  else:
    tf_logits = lowering.export_to_tf_tensor(fully_replicated_logits)

  with mtf.utils.outside_all_rewrites():
    if mode == tf.estimator.ModeKeys.TRAIN:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.TRAIN,
          loss=tf_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(tf_logits):
        mean_logits = tf.metrics.mean(tf_logits)
        return {'mean_logits': mean_logits}

      eval_metrics = (metric_fn, [tf_logits])
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          loss=tf_loss,
          eval_metrics=eval_metrics)


def run_toy_model_gpu():
  """Run a toy model on GPU."""

  iterations_per_loop = FLAGS.iterations
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  classifier = estimator_lib.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)
  current_step = estimator_lib._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
  logging.info('Current step %d', current_step)
  if FLAGS.steps_per_checkpoint == 0:
    classifier.train(input_fn=ToyModelInput(), max_steps=FLAGS.train_steps)
    return
  while current_step < FLAGS.train_steps:
    next_checkpoint = min(current_step + FLAGS.steps_per_checkpoint,
                          FLAGS.train_steps)
    classifier.train(input_fn=ToyModelInput(), max_steps=next_checkpoint)
    current_step = next_checkpoint
    logging.info('Starting to evaluate.')
    eval_results = classifier.evaluate(
        input_fn=ToyModelInput(),
        steps=156)  # since we have 10000 examples and batch_size = 64 per host
    logging.info('Eval results: %s', eval_results)


def main(_):
  run_toy_model_gpu()


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
