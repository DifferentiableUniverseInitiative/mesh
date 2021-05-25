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
  def __init__(self, x, message=None, domain_name=None, name=None):
    super(NVTXOperation, self).__init__([x], x.mesh, name=name or "nvtx")
    self._outputs = [
        mtf.Tensor(self, x.shape, x.dtype)
    ]
    self.message = message
    self.domain_name = domain_name

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self) 

    x = lowering.tensors[self.inputs[0]]
    
    def _tf_fn(x):
        x, nvtx_context = nvtx_tf.ops.start(x, message=self.message,
                                            domain_name=self.domain_name)
        x = nvtx_tf.ops.end(x, nvtx_context)       
        return x
    value = mesh_impl.slicewise(_tf_fn, x)
    lowering.set_tensor_lowering(self.outputs[0], value)


def add_nvtx(x, message=None, domain_name=None,  name=None):
  return NVTXOperation(x, message, domain_name, name).outputs[0]