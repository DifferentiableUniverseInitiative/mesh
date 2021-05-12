from mpi4py import MPI
import os
# Pin only one GPU per horovod process
comm = MPI.COMM_WORLD
#os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(comm.rank+1) # This is specific to my machine

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf
from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl

# We create a small mesh
graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")

batch_dim = mtf.Dimension("batch", 2)
nx_dim = mtf.Dimension('nx_block', 2)
ny_dim = mtf.Dimension('ny_block', 1)

# Defines the mesh structure
mesh_shape = [ ("row", 1), ("col", 2)]
layout_rules = [('ny_block', 'row'),  ("nx_block","col")]

data = mtf.ones(mesh, [batch_dim, ny_dim, nx_dim])

toto = mtf.reduce_sum(data, reduced_dim=ny_dim)
toti = mtf.reduce_sum(toto, reduced_dim=nx_dim)

mesh_impl = HvdSimdMeshImpl(mtf.convert_to_shape(mesh_shape), 
                            mtf.convert_to_layout_rules(layout_rules))

lowering = mtf.Lowering(graph, {mesh:mesh_impl})

res = lowering.export_to_tf_tensor(toti)

# Execute and retrieve result
with tf.Session() as sess:
    r = sess.run(res)

print("Final result", r)
