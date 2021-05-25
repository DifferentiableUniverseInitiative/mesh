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

"""SIMD Mesh implementation using an Horovod backend (for GPU clusters)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import collections
import numpy as np

from mesh_tensorflow import ops_with_redefined_builtins as mtf
from mesh_tensorflow import utils
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow.compat.v1 as tf
from mpi4py import MPI
import horovod.tensorflow as hvd

class HvdSimdMeshImpl(mtf.MeshImpl):
  """Mesh implementation for GPU using SIMD and MPI operations."""

  def __init__(self,
               shape,
               layout):
    """Create a HvdSimdMeshImpl.

    Args:
      shape: an input to mtf.convert_to_shape()
      layout: an input to mtf.convert_to_layout_rules()
    """
    super(HvdSimdMeshImpl, self).__init__(shape, layout)
    tf.logging.info("HvdSimdMeshImpl init: {0} {1}".format(shape, layout))

    # Initializing communicators
    comms, comms_id = self._create_communicators(shape)
    self._comms = comms
    self._comms_id = comms_id

    # And initializing horovod with our set of communicators
    hvd.init(comm=[c for _, c in comms.items()])
    
    self.graph_device_function_stacks = []
    self.copy_master_to_slice_ops = []

  def _create_communicators(self, mesh_shape):
    """ Creates MPI communicators required for a given mesh shape
    Note that the first communicator, of index 0 is the world comm.
    """
    dims = [s.size for s in mesh_shape]
    cart_comm = MPI.COMM_WORLD.Create_cart(dims=dims,
                                           periods=[False]*len(dims))
    communicators = collections.OrderedDict()
    communicators_id = collections.OrderedDict()
    communicators["world"] = cart_comm
    communicators_id["world"] = 0
    # Extract one sub communicator per dimension
    for i,s in enumerate(mesh_shape):
        remain_dims = [False]*len(dims)
        remain_dims[i] = True
        communicators[s.name] = cart_comm.Sub(remain_dims)
        communicators_id[s.name] = i+1
    return communicators, communicators_id

  class LaidOutTensor(object):
    """One Slice."""

    def __init__(self, tensor_list):
      assert isinstance(tensor_list, list)
      self._tensor_list = tensor_list

    def __repr__(self):
      return "[" + ",".join([str(t) for t in self._tensor_list]) + "]"

    @property
    def tensor_list(self):
      return self._tensor_list

    @property
    def one_slice(self):
      return self._tensor_list[0]

    @classmethod
    def from_tensor_list(cls, tensor_list):
      return cls(tensor_list)

    @property
    def all_slices(self):
      return self._tensor_list

    @property
    def slice_shape(self):
      return self.one_slice.shape.as_list()

    def to_laid_out_tensor(self):
      return self

  class LaidOutVariable(object):
    """Maintains slice-variables and copy operations."""

    def __init__(self, variable, mesh_impl):
      """Create a LaidOutVariable.

      Args:
        variable: a Variable (Operation)
        mesh_impl: a MeshImpl
      """
      self._variable = variable
      self._mesh_impl = mesh_impl
      shape = variable.outputs[0].shape
      slice_shape = mesh_impl.slice_shape(shape)
      base_name = variable.name
      slices = []
      slices_with_master_dtype = []
      with tf.device(variable.master_device), utils.outside_all_rewrites():
        zero_tensor = tf.zeros(slice_shape, dtype=variable.slice_dtype)

      # pylint: disable=protected-access
      init_device_stack = tf.get_default_graph()._device_function_stack

      for physical_pnum in xrange(mesh_impl.size):
        slice_var_name = base_name + "_slice_%d" % physical_pnum
        # Use tf.Variable instead of tf.get_variable since latter adds lots of
        # useless operations to the TF graph. Use tf.get_variable only if
        # in a AUTO_REUSE scope.
        # Note: Repeatedly 'with tf.device():' slows down the graph
        # construction. Therefore we directly use the cached device_stack here.
        tf.get_default_graph()._device_function_stack = (
            mesh_impl.graph_device_function_stacks[physical_pnum])

        if tf.get_variable_scope().reuse == tf.AUTO_REUSE:
          slice_var = tf.get_variable(
              initializer=zero_tensor,
              trainable=self._variable.trainable,
              collections=["TPU_VAR"],
              dtype=variable.slice_dtype,
              name=slice_var_name)
        else:
          slice_var = tf.Variable(
              initial_value=zero_tensor,
              trainable=self._variable.trainable,
              collections=["TPU_VAR"],
              dtype=variable.slice_dtype,
              name=slice_var_name,
              expected_shape=slice_shape)

        slices.append(slice_var)

      # Restore the initial stack
      tf.get_default_graph()._device_function_stack = init_device_stack
      # pylint: enable=protected-access

      self._laid_out_tensor = mesh_impl.LaidOutTensor(
          [tpu_variables.ReplicatedVariable(base_name, slices)])
      with tf.device(variable.master_device), utils.outside_all_rewrites():
        if os.environ.get("MTF_SEQUENCE_MODE", "") == "1":
          if mesh_impl.copy_master_to_slice_ops:
            with tf.control_dependencies(
                [mesh_impl.copy_master_to_slice_ops[-1]]):
              self._copy_master_to_slices = self._gen_copy_master_to_slices_op(
                  variable.get_master(), shape, slices, slice_shape)
          else:
            self._copy_master_to_slices = self._gen_copy_master_to_slices_op(
                variable.get_master(), shape, slices, slice_shape)

          mesh_impl.copy_master_to_slice_ops.append(self._copy_master_to_slices)
        else:
          self._copy_master_to_slices = self._gen_copy_master_to_slices_op(
              variable.get_master(), shape, slices, slice_shape)
        slices_with_master_dtype = [
            tf.cast(s, variable.master_dtype) for s in slices]
        slices_with_master_dtype = [
            slices_with_master_dtype[mesh_impl.l2p(logical_pnum)]
            for logical_pnum in range(mesh_impl.size)]
        self._copy_slices_to_master = variable.assign_to_master(
            mesh_impl.combine_slices(slices_with_master_dtype, shape,
                                     device=variable.master_device))

    def _gen_copy_master_to_slices_op(self, master_variable, master_shape,
                                      slices, slice_shape):
      """Generate ops which slices master and assign to slices.

      Args:
        master_variable: The master variable.
        master_shape: The shape of master variable.
        slices: The list of slice-variables in physical order.
        slice_shape: The shape of the slice variable.
      Returns:
        A grouped tf.assign ops.
      """
      mesh_impl = self._mesh_impl
      master_layout = mesh_impl.tensor_layout(master_shape)
      # For handling case: master is float32 and slices are bfloat16.
      if master_variable.dtype != slices[0].dtype:
        master_variable = tf.cast(master_variable, slices[0].dtype)
      assign_ops = []
      if master_layout.is_fully_replicated:
        assign_ops = [tf.assign(t, master_variable) for t in slices]
      else:
        slice_dict = {}
        for logical_pnum in xrange(len(slices)):
          slice_begin = mesh_impl.slice_begin(master_shape, logical_pnum)
          slice_begin_tuple = tuple(slice_begin)
          # Reuse the same slice if slice_begin doesn't change.
          if slice_begin_tuple not in slice_dict:
            slice_dict[slice_begin_tuple] = tf.slice(master_variable,
                                                     slice_begin, slice_shape)
          physical_pnum = mesh_impl.l2p(logical_pnum)
          assign_ops.append(
              tf.assign(slices[physical_pnum], slice_dict[slice_begin_tuple]))
      return tf.group(assign_ops)

    def assign_to_slices(self, assign_fn, values, assign_to_tensor_list=None):
      """Assign to the slice variables.

      Args:
        assign_fn: a function from
          (mtf.Variable, tf.Variable, tf.Tensor) -> tf.Operation
        values: a list of tf.Tensor
        assign_to_tensor_list: an optional list of tf.Variable

      Returns:
        a tf.operation
      """
      if assign_to_tensor_list is None:
        assign_to_tensor_list = self._laid_out_tensor.all_slices
      # Handle both N -> 1 and N -> N cases.
      num_slices = min(len(assign_to_tensor_list), len(values))
      devices = [""] * num_slices
      return tf.group(
          mtf.parallel(devices, assign_fn,
                       [self._variable] * len(devices),
                       assign_to_tensor_list[:num_slices],
                       values[:num_slices]))

    @property
    def laid_out_tensor(self):
      return self._laid_out_tensor

    @property
    def copy_master_to_slices(self):
      return self._copy_master_to_slices

    @property
    def copy_slices_to_master(self):
      return self._copy_slices_to_master

  def laid_out_pnum(self):
    """Returns a LaidOutTensor containing the logical processor number.

    Returns:
      a LaidOutTensor where each slice is an integer scalar
    """
    return self.LaidOutTensor([self.pnum_tensor])

  def allreduce(self, x, mesh_axes, reduction_fn_string):
    """Grouped allreduce, (summed across the given dimensions).

    Args:
      x: a LaidOutTensor
      mesh_axes: a list of integers
      reduction_fn_string: "SUM"
    Returns:
      a LaidOutTensor
    Raises:
      ValueError: if the reduction is not yet implemented.
    """
    if not mesh_axes:
      return x

    if reduction_fn_string != 'SUM':
      #TODO: Either add additional reduction ops to Horovod
      # or adapt the TPU mechanism for different types of reduce
      raise ValueError("Only sum reduction is implemented with horovod backend")

    x = x.to_laid_out_tensor()
    t = x.one_slice

    # In case the tensor is complex, let's split it in real and imag parts
    is_complex = t.dtype == tf.complex64
    if is_complex:
      t = tf.stack([tf.math.real(t), tf.math.imag(t)], axis=-1)

    # Performing reduce operation for all axes
    for mesh_axis in mesh_axes:
      s = self.shape[mesh_axis]
      t = hvd._allreduce(t, communicator_id=self._comms_id[s.name])

    if is_complex:
      t = tf.complex(t[...,0], t[...,1])

    return self.LaidOutTensor([t])

  def allconcat(self, x, mesh_axis, concat_axis, stack=False):
    """Grouped allconcat (like MPI allgather followed by concat).

    TODO(noam): inefficient - replace with a XLA allconcat when available

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer - the mesh axis along which to group
      concat_axis: an integer (the Tensor axis along which to concatenate)
      stack: a boolean - whether to atack instead of concat
    Returns:
      a LaidOutTensor
    """
    x = x.to_laid_out_tensor()
    t = x.one_slice
    old_shape = t.shape.as_list()
    num_parts = self.shape[mesh_axis].size
    name_dim  = self.shape[mesh_axis].name

    print('CONCAT', stack)

    # Moving axis to concatenate at the top
    perm = [concat_axis] + [i for i in range(len(old_shape)) if i not in [concat_axis]]
    if not stack:
      t = tf.transpose(t, perm)

    # Horovod allconcat only support concatenations over the first dimension
    # TODO: add horovod tool to directly concatenate on given axis
    t = tf.expand_dims(t, 0)

    # Performing concatenation on first axis
    # In case the tensor is complex, let's split it in real and imag parts
    is_complex = t.dtype == tf.complex64
    if is_complex:
      t = tf.stack([tf.math.real(t), tf.math.imag(t)], axis=-1)

    t = hvd.allgather(t, communicator_id=self._comms_id[name_dim])

    if is_complex:
      t = tf.complex(t[...,0], t[...,1])
    
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(inv), dtype=np.int32)

    if not stack:
      t = tf.reshape(t, [old_shape[concat_axis]*num_parts]+t.shape.as_list()[2:])
      t = tf.transpose(t, inv)
    else:
      t = tf.tranpose(t, [i+1 if i < concat_axis else \
                         i if i > concat_axis else 0 \
                         for i in range(len(old_shape)+1)])
    
    # # Moving concatenated dimension to concat_axis
    
    # t = tf.transpose(t, [i+1 if i < concat_axis-1 else \
    #                      i if i > concat_axis-1 else 0 \
    #                      for i in range(len(old_shape)+1)])

    # # print('inds', [i+1 if i < concat_axis-1 else \
    # #                      i if i > concat_axis-1 else 0 \
    # #                      for i in range(len(old_shape)+1)])
    # if not stack:
    #   new_shape = old_shape[:]
    #   new_shape[concat_axis] *= num_parts
    #   t = tf.reshape(t, new_shape)

    # # Let's just make sure everybody got there.
    # hvd.join()
    return self.LaidOutTensor([t])

  def alltoall(self, x, mesh_axis, split_axis, concat_axis):
    """Grouped alltoall (like MPI alltoall with splitting and concatenation).

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer the mesh axis along which to group
      split_axis: an integer (the Tensor axis along which to split)
      concat_axis: an integer (the Tensor axis along which to concatenate)
    Returns:
      a LaidOutTensor
    """
    x = x.to_laid_out_tensor()
    t = x.one_slice
    old_shape = t.shape.as_list()
    name_dim  = self.shape[mesh_axis].name

    perm = [split_axis, concat_axis] + [i for i in range(len(old_shape)) if i not in [split_axis, concat_axis]]
    
    # Ok, so horovod ops happen on the first axis, we need to transpose split
    # axis to the front of the tensor
    # t = tf.transpose(t, [split_axis] + [i if i < split_axis else i+1 \
    #                                     for i in range(len(old_shape)-1)])
    t = tf.transpose(t, perm)

    is_complex = t.dtype == tf.complex64
    if is_complex:
      t = tf.stack([tf.math.real(t), tf.math.imag(t)], axis=-1)

    # The all2all should preserve the shape of the tensor, so we manually
    # reshape the tensor after all2all
    s = t.shape.as_list()
    n = hvd.size(communicator_id=self._comms_id[name_dim])
    # Now we apply an all2all on this first dimension
    t = hvd.alltoall(t, communicator_id=self._comms_id[name_dim])
    t = tf.reshape(t, [n, s[0]//n]+ s[1:])
    
    if is_complex:
      t = tf.complex(t[...,0], t[...,1])

    # At this stage t looks like [n, split_axis/n, concat_axis, ...]
    t = tf.transpose(t, [1,0]+[i+2 for i in range(len(old_shape)-1)])
    # At this stage t looks like [split_axis/n, n, concat_axis, ...]
    t = tf.reshape(t, [s[0]//n, s[1]*n]+s[2:-1])
    # At this stage t looks like [split_axis/n, n*concat_axis, ...]
    # We can just invert the original permutation
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(inv), dtype=np.int32)
    print("Heel", perm, inv, n)
    t = tf.transpose(t, inv)

    # # Permutation to move the split axis back where it belongs

    # perm = [0, conc]
    # perm = [i+1 if i < split_axis else \
    #                      i if i > split_axis else 0 \
    #                      for i in range(len(old_shape))]

    # perm = [i for i in range(len(old_shape))]
    # np.swapaxes(perm, 1, )
    # t = tf.transpose(t, [0, ])


    # # Moving concatenated dimension to concat_axis
    # t = tf.transpose(t, [i+1 if i < concat_axis else \
    #                      i if i > concat_axis else 0 \
    #                      for i in range(len(old_shape))])
    # print([split_axis] + [i if i < split_axis else i+1 \
    #                                     for i in range(len(old_shape)-1)])
    # print([i+1 if i < concat_axis else \
    #                      i if i > concat_axis else 0 \
    #                      for i in range(len(old_shape))])
    # print("Yo",old_shape, split_axis, concat_axis, t.shape)
    x = self.LaidOutTensor([t])
    return x

  # def receive(self, x, mesh_axis, source_pcoord):
  #   """Collective receive in groups.
  #
  #   Each group contains the processors that differ only in mesh_axis.
  #
  #   ```python
  #   group_size = self.shape[mesh_axis].size
  #   ```
  #
  #   Args:
  #     x: a LaidOutTensor
  #     mesh_axis: an integer
  #     source_pcoord: a list of optional integers. Each element is either None
  #       or an integer in [0, group_size). If source_pcoord[k] is None, then the
  #       output for the k-th processor in each group is a zero tensor. If
  #       source_pcoord[k] is not None, then the output for the k-th processor in
  #       each group is equal to the input for the source_pcoord[k]-th processor
  #       in that group.
  #
  #   Returns:
  #     a LaidOutTensor
  #   """
  #   raise NotImplementedError
  # x = x.to_laid_out_tensor()
  # t = x.one_slice
  # source_target_pairs = []
  #
  # for pnum in xrange(self.size):
  #   coord = mtf.pnum_to_processor_coordinates(self.shape, pnum)
  #   k = coord[mesh_axis]
  #   if source_pcoord[k] is not None:
  #     coord[mesh_axis] = source_pcoord[k]
  #     source_pnum = mtf.processor_coordinates_to_pnum(self.shape, coord)
  #     source_target_pairs.append(
  #         [self.l2p(source_pnum),
  #          self.l2p(pnum)])
  #
  # if not source_target_pairs:
  #   ret = tf.zeros_like(t, t.dtype)
  # elif t.dtype in [tf.float32, tf.bfloat16, tf.int32]:
  #   ret = tpu_ops.collective_permute(t, source_target_pairs)
  # else:
  #   # If t is not one of the allowed types, cast and cast back.
  #   ret = tf.cast(tpu_ops.collective_permute(
  #       tf.cast(t, tf.float32), source_target_pairs), t.dtype)
  #
  # return self.LaidOutTensor([ret])

  def shift_by_n_processors(self, x, mesh_axis, offset, wrap):
    """Receive the slice from processor pcoord - offset.

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer
      offset: an integer
      wrap: a boolean. If True, then wrap around. Otherwise, pad with zeros.

    Returns:
      a LaidOutTensor
    """
    x = x.to_laid_out_tensor()
    t = x.one_slice
    n = self.shape[mesh_axis].size
    name_dim  = self.shape[mesh_axis].name

    # Because we don't have access to a generic send-recv or shift operation...
    # we are going to do something very suboptimal: gather all slices and only
    # keeping the one we want locally
    t = tf.expand_dims(t, 0)

    # In case the tensor is complex, let's split it in real and imag parts
    is_complex = t.dtype == tf.complex64
    if is_complex:
      t = tf.stack([tf.math.real(t), tf.math.imag(t)], axis=-1)

    t = hvd.allgather(t, communicator_id=self._comms_id[name_dim])

    if is_complex:
      t = tf.complex(t[...,0], t[...,1])    

    # and....we only need to keep one slice... but which one...
    c = hvd.rank(communicator_id=self._comms_id[name_dim]) + offset
    if ((c >= n) or (c <0)) and (not wrap):
      t = tf.zeros_like(x.one_slice)
    else:
      t = t[c % n]
    print("HEEELLOO")
    return self.LaidOutTensor([t])

  def slice_begin(self, tensor_shape):
    """Variant of slice begin for SIMD"""
    tensor_layout = self.tensor_layout(tensor_shape)
    slice_shape = self.slice_shape(tensor_shape)

    slice_begins = [
      0 if mesh_axis is None else hvd.rank(communicator_id=self._comms_id[self.shape[mesh_axis].name])*slice_shape[i]
      for i,mesh_axis in enumerate(tensor_layout)
      ]
    return slice_begins

  def slice(self, tf_tensor, tensor_shape):
    """"Slice out the corresponding part of tensor."""
    tensor_layout = self.tensor_layout(tensor_shape)
    if tensor_layout.is_fully_replicated:
      return self.LaidOutTensor([tf_tensor])
    else:
      slice_shape = self.slice_shape(tensor_shape)
      slice_begins = self.slice_begin(tensor_shape)
      slice_begins_tensor = tf.stack(slice_begins)
      return self.LaidOutTensor(
          [tf.slice(tf_tensor, slice_begins_tensor, slice_shape)])

  def slicewise(self, fn, *inputs):
    """Execute a function in parallel on all slices.

    Args:
      fn: a function from tf.Tensors to tf.Tensor or a tuple of tf.Tensors.
      *inputs: a list of inputs.  Each input is either a LaidOutTensor or
        is convertible to a tf.Tensor.
    Returns:
      a LaidOutTensor, or a tuple of LaidOutTensors if fn returns a tuple.
    """
    # convert all inputs to LaidOutTensor where possible
    inputs = mtf.convert_args_to_laid_out_tensors(inputs)
    ret = fn(*[
        x.one_slice if isinstance(x, self.LaidOutTensor) else x
        for x in inputs])
    if isinstance(ret, tuple):
      return tuple([self.LaidOutTensor([t]) for t in ret])
    else:
      return self.LaidOutTensor([ret])

  def random(self, shape, tf_fn, kwargs):
    """Call a random tf operation (e.g. random_uniform).

    Args:
      shape: a Shape
      tf_fn: a function such as tf.random.uniform
      kwargs: kwargs to pass to tf_fn, including the seed

    Returns:
      a LaidOutTensor
    """
    """
    # Old implementation
    slice_shape = self.slice_shape(shape)

    x = tf_fn(slice_shape, **kwargs)
    # TPU does not have seeds enabled.  Sync up the
    # random choices by zeroing out all but the first core per group of
    # identical slices, then allreducing by group.
    layout = self.tensor_layout(shape)
    # we need to sync across these axes.
    mesh_axes = [i for i in xrange(self.ndims)
                 if i not in layout.tensor_axis_to_mesh_axis]
    multiplier = 1.0
    for axis in mesh_axes:
      multiplier *= tf.cast(
          tf.equal(self.laid_out_pcoord(axis).one_slice, 0), x.dtype)
    x *= multiplier
    x = self.LaidOutTensor([x])
    x = self.allreduce(x, mesh_axes, "SUM")
    return x
    """
    # Tobi-doc
    # kwargs['seed'] is the original seed.
    # If we don't distribute we use that same seed for all the random variables
    # If we distribute a big tensor we need to change the seed.
    # Dependign on the mesh, the tensor dimensions, and how we are distribting them we need to generate a set of seeds for each
    # GPU. These seeds might need to be identical or different as a function of the distribution strategy.
    
    slice_shape = self.slice_shape(shape)
    # Get the biggest dimension to be sure that the seeds are different when they are supposed to be different.
    max_dim = np.max(slice_shape)
    # To check if the tensors are fully replicated and then to check on which dimensions they are disbributed.
    tensor_layout = self.tensor_layout(shape)
   
    if self.tensor_layout(shape).is_fully_replicated:
        # Same seed for everyone
        x = tf_fn(slice_shape, **kwargs)
        return self.LaidOutTensor([x])
    
    else:
        # We need difference seeds.
        slice_begins = [
          0 if mesh_axis is None else hvd.rank(communicator_id=self._comms_id[self.shape[mesh_axis].name])*slice_shape[i]
          for i,mesh_axis in enumerate(tensor_layout)
          ]
        #  print('slice_begins: ', slice_begins, ', hvd.rank(): ', hvd.rank())

        my_seed = np.sum([0 if dim_id==0 else it*max_dim+dim_id  for it, dim_id in enumerate(slice_begins)])
        # print('hvd.rank(): ', hvd.rank(), ', my_seed: ', my_seed)

        if 'seed' in kwargs.keys():
          kwargs['seed'] += my_seed
          # print("kwargs['seed']", kwargs['seed'])
        else:
          print('WARNING! Setting default seed to 0.')
          kwargs['seed'] = 0
          kwargs['seed'] += my_seed
          # print("kwargs['seed']", kwargs['seed'])

 
        x = tf_fn(slice_shape, **kwargs)
        return self.LaidOutTensor([x])
    

  def export_to_tf_tensor(self, x, laid_out_x):
    """Turn a Tensor into a tf.Tensor.

    Args:
      x: a Tensor
      laid_out_x: a LaidOutTensor
    Returns:
      a tf.Tensor
    """
    tensor_layout = self.tensor_layout(x.shape)
    if not tensor_layout.is_fully_replicated:
      print("Warning: Exported tensor is not fully replicated"
            " x.shape = %s tensor_layout=%s"
            % (x.shape, tensor_layout))
      # raise NotImplementedError(
      #     "SimdMeshImpl only supports export_to_tf_tensor of fully-replicated "
      #     "Tensors.  Try reshaping to new dimension names. "
      #     " x.shape = %s tensor_layout=%s"
      #     % (x.shape, tensor_layout))
    return laid_out_x.one_slice

  def import_tf_tensor(self, x, tf_x):
    """Import a tf.Tensor, producing a LaidOutTensor.

    Args:
      x: a Tensor
      tf_x: a tf.Tensor
    Returns:
      a LaidOutTensor
    """
    return self.slice(tf_x, x.shape)

  @property
  def supports_control_dependencies(self):
    return False

  def einsum(self, equation, *slices):
    """Override this for custom einsum implementation.

    Args:
      equation: a string
      *slices: a list of tf.Tensor
    Returns:
      a tf.Tensor
    """
    return tf.einsum(equation, *slices)
