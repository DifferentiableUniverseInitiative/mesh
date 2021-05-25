"""Required Additional Mesh TensorFlow ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import numpy as np
import nvtx.plugins.tf as nvtx_tf

class NVTXOperation(mtf.Operation):
  """Mesh implementation of nvtx tracers"""
  def __init__(self, x, name=None):
    super(NVTXOperation, self).__init__([x], x.mesh, name=name or "nvtx")
    self._outputs = [
        mtf.Tensor(self, x.shape, x.dtype)
    ]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self) 

    x = lowering.tensors[self.inputs[0]]
    
    # have it go through NVTX
    x, nvtx_context = nvtx_tf.ops.start(x, message='Dense 1',
        grad_message='Dense 1 grad', domain_name='Forward')

    x = nvtx_tf.ops.end(x, nvtx_context)
    
    lowering.set_tensor_lowering(self.outputs[0], x)


def add_nvtx(x, name=None):
  return NVTXOperation(x, name).outputs[0]