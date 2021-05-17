from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from mpi4py import MPI
#import os
# Pin only one GPU per horovod process
#comm = MPI.COMM_WORLD
#os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(comm.rank+1) # This is specific to my machine

import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import flowpm.mesh_ops as mpm
from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl

from matplotlib import pyplot as plt

# Options needed for the Device Placement Mesh Implementation
tf.flags.DEFINE_integer("gpus_per_node", 4, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 4, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")
tf.flags.DEFINE_integer("cube_size", 128, "Size of the 3D volume.")
tf.flags.DEFINE_integer("hsize", 0, "halo size")
tf.flags.DEFINE_integer("batch_size", 1, "Mini-batch size for the training. Note that this"
                        "is the global batch size and not the per-shard batch.")
#tf.flags.DEFINE_string("mesh_shape", "b1:4,b2:4", "mesh shape")
#tf.flags.DEFINE_string("layout", "nx:b1,tny:b1,ny:b2,tnz:b2", "layout rules")

#mesh flags
tf.flags.DEFINE_integer("nx", 4, "# blocks along x")
tf.flags.DEFINE_integer("ny", 4, "# blocks along y")


FLAGS = tf.flags.FLAGS

#batch_dim = mtf.Dimension("batch", 2)
#nx_dim = mtf.Dimension('nx_block', 2)
#ny_dim = mtf.Dimension('ny_block', 1)


#data = mtf.ones(mesh, [batch_dim, ny_dim, nx_dim])

#toto = mtf.reduce_sum(data, reduced_dim=ny_dim)
#toti = mtf.reduce_sum(toto, reduced_dim=nx_dim)

def benchmark_model(mesh):
  """
  Initializes a 3D volume with random noise, and execute a forward FFT
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  x_dim = mtf.Dimension("nx", FLAGS.cube_size)
  y_dim = mtf.Dimension("ny", FLAGS.cube_size)
  z_dim = mtf.Dimension("nz", FLAGS.cube_size)

  tx_dim = mtf.Dimension("tnx", FLAGS.cube_size)
  ty_dim = mtf.Dimension("tny", FLAGS.cube_size)
  tz_dim = mtf.Dimension("tnz", FLAGS.cube_size)

  # Create field
  field = mtf.random_normal(mesh, [batch_dim, x_dim, y_dim, z_dim])

  input_field = field
  field = mtf.cast(field, tf.complex64)
  err = 0

  # Performs a back and forth FFTs in the same session
  
  # Apply FFT
  fft_field = mpm.fft3d(field, [tx_dim, ty_dim, tz_dim])

  # Inverse FFT
  field = mpm.ifft3d(fft_field * 1, [x_dim, y_dim, z_dim])
  
  field = mtf.cast(field, tf.float32)
  
  # Compute the residuals between inputs and outputs
  err += mtf.reduce_sum(mtf.abs(field - input_field))
  
  return err, input_field, field

def main(_):

  # Defines the mesh structure
  #mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape) #mesh_shape = [ ("row", 1), ("col", 2)]
  #layout_rules = mtf.convert_to_layout_rules(FLAGS.layout) #layout_rules = [('ny_block', 'row'),  ("nx_block","col")]

  mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]
  mesh_shape = mtf.convert_to_shape(mesh_shape)
  layout_rules = [("nx", "row"), ("ny", "col"), 
                  ("tny", "row"), ("tnz", "col")]
  layout_rules = mtf.convert_to_layout_rules(layout_rules)

  # Instantiate the mesh impl
  mesh_impl = HvdSimdMeshImpl(mesh_shape,layout_rules)
 
  # Create the mesh
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")

  # Build the model
  fft_err, input_field, output_field = benchmark_model(mesh)
  
  lowering = mtf.Lowering(graph, {mesh:mesh_impl})

  # Retrieve output of computation
  res = lowering.export_to_tf_tensor(fft_err)
  in_field = lowering.export_to_tf_tensor(input_field)
  out_field = lowering.export_to_tf_tensor(output_field)

  # Execute and retrieve result
  with tf.Session() as sess:
    r, a, c = sess.run([res, in_field, out_field])

    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.imshow(a[0].sum(axis=2))
    plt.title('Initial Field')

    plt.subplot(122)
    plt.imshow(c[0].sum(axis=2))
    plt.title('Forward-backward 3D FFT')
    #plt.colorbar()
    plt.savefig("mesh_nbody_%d-row:%d-col:%d.png" %
              (FLAGS.cube_size, FLAGS.nx, FLAGS.ny))
    plt.close()
  exit(-1)

  print("Final result", r)

if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
