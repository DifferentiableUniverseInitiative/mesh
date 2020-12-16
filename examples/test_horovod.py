import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf
import horovod.tensorflow as hvd
from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl

# Horovod: initialize Horovod.
hvd.init()

# Check number of processes and rank
print("My rank is ", hvd.rank(), "out of ", hvd.size())

# We create a small mesh
graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")

batch_dim = mtf.Dimension("batch", 2)
# nd_nx_dim = mtf.Dimension("no_distributedx", 2)
# nd_ny_dim = mtf.Dimension("no_distributedy", 2)
nx_dim = mtf.Dimension('nx_block', 2)
ny_dim = mtf.Dimension('ny_block', 2)

# Defines the mesh structure
mesh_shape = [ ("row", 2), ("col", 2)]
layout_rules = [('ny_block', 'row'),  ("nx_block","col")]

data = mtf.ones(mesh, [batch_dim, ny_dim, nx_dim])

#toto = mtf.reshape(data, [batch_dim, nd_nx_dim, nd_ny_dim])

#toti = mtf.reshape(toto, [batch_dim, nd_nx_dim, nd_ny_dim])

toto = mtf.reduce_sum(data, reduced_dim=ny_dim)
toti = mtf.reduce_sum(toto, reduced_dim=nx_dim)

#toti = mtf.reduce_sum(toto, reduced_dim=nx_dim) #mtf.reshape(toto, [batch_dim, nd_nx_dim])

mesh_impl = HvdSimdMeshImpl(mesh_shape, layout_rules)

lowering = mtf.Lowering(graph, {mesh:mesh_impl})

result = lowering.export_to_tf_tensor(toti)

# Execute and retrieve result
with tf.Session() as sess:
    fin_ref = sess.run(result)

print("Final result", fin_ref)
