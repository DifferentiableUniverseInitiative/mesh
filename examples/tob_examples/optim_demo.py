import numpy as np
import os
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf
from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl

tf.flags.DEFINE_integer("gpus_per_node", 4, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 4, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")
tf.flags.DEFINE_integer("nx", 2, "number of slices along x dim")
tf.flags.DEFINE_integer("ny", 2, "number of slices along x dim")
tf.flags.DEFINE_integer("nz", 1, "number of slices along z dim")
tf.flags.DEFINE_integer("nc", 8, "Size of data cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
FLAGS = tf.flags.FLAGS

def new_model_fn():


    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    # Define dimensions
    batch_dim = mtf.Dimension("batch", FLAGS.batch_size)

    nc_x_dim = mtf.Dimension('nc_x_dim', FLAGS.nc)
    nc_y_dim = mtf.Dimension('nc_y_dim', FLAGS.nc)
    nc_z_dim = mtf.Dimension('nc_z_dim', 1)

    hidden_dim_int = 16
    hidden_dim  = mtf.Dimension('h', hidden_dim_int)

    # Define the input
    # mtf_input = mtf.import_tf_tensor(mesh, im_input, shape=mtf.Shape([batch_dim, nc_x_dim, nc_y_dim, nc_z_dim])) 

    mtf_input = mtf.random_uniform(mesh, [batch_dim, nc_x_dim, nc_y_dim, nc_z_dim])

    # Define the network
    net_out = mtf.layers.dense(mtf_input, hidden_dim) 
    net_out = mtf.reduce_sum(net_out, output_shape=[batch_dim, hidden_dim])

    # Define the loss
    one_tensor = mtf.import_tf_tensor(
            mesh,
            tf.ones([FLAGS.batch_size, hidden_dim_int]),
            shape=mtf.Shape([batch_dim, hidden_dim]))
    
    loss = mtf.reduce_sum(mtf.square(net_out - one_tensor))
    # loss = mtf.reduce_sum(mtf.square(net_out))

    # return net_out, loss
    return loss


def main(_):
    # NEW
    global_step = tf.train.get_global_step()
    
    # Defines the mesh structure
    mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]
    # layout_rules = [("nx_block","row"), ("ny_block","col")]
    layout_rules = [("nc_x_dim","row"), ("nc_y_dim","col")]
    
    #mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, devices)
    mesh_impl = HvdSimdMeshImpl(mtf.convert_to_shape(mesh_shape), 
                                mtf.convert_to_layout_rules(layout_rules))
    
    
    # Create computational graphs
    net  = new_model_fn()
    # Lower mesh computation
    graph = net.graph
    mesh = net.mesh
    # Retrieve output of computation
    # result = lowering.export_to_tf_tensor(net)
    
    var_grads = mtf.gradients(
        [net], [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf.optimize.AdafactorOptimizer()
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
   
    print('update_ops: ',update_ops)

    lowering = mtf.Lowering(graph, {mesh:mesh_impl})
    # Perform some last processing in normal tensorflow
    result = lowering.export_to_tf_tensor(net)
    out = tf.reduce_mean(result)

    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    train_op = tf.group(tf_update_ops)


    print('Im about to enter the session..')
    print('before session: ', tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            r, _ = sess.run([out, train_op])
            print('iter', i)
            print(sess.run(tf.trainable_variables()))
        # r = sess.run(out)
    
    print("output of computation", r)
    exit(0)


if __name__ == "__main__":
  tf.app.run(main=main)



