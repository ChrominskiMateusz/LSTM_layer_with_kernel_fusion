import tensorflow as tf
import time as time
from tensorflow.python.client import timeline

sess = tf.InteractiveSession()

# num_units = 128
# batch_size = 128 * 100
# input_size = 28

a = [50000, 50000]
b = [25, 50000]

# Init with floats from 0 to 1
x = tf.random_uniform(a, minval=0, maxval=1, dtype=tf.float32)                # icfo A
y = tf.random_uniform(b, minval=0, maxval=1, dtype=tf.float32)   # w    B
y_t = tf.transpose(y)                                                                                   # transpose w

# k - number of largest elements that will remain in each row
k = 500
values, indices = tf.nn.top_k(x, k, sorted=False)
values = tf.reshape(values, [-1])

# print(x.eval())
# print(values.eval())
# print(indices.eval())

my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
my_range_repeated = tf.tile(my_range, [1, k])

full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], axis=2)
full_indices = tf.reshape(full_indices, [-1, 2])
#print(full_indices.eval())

# Allocate SparseTensor
sparse_x = tf.SparseTensor(indices=tf.cast(full_indices, dtype=tf.int64), values=values, dense_shape=x.shape)
#print(sparse_x.eval())

# Usage of standard Tensorflow matrix multiplication
resStandardMatMul = tf.matmul(x, y, transpose_b=True)
resSparseMatMul0 = tf.sparse.sparse_dense_matmul(sparse_x, tf.transpose(y))
resSparseMatMul1 = tf.sparse.sparse_dense_matmul(sparse_x, y, adjoint_b=True)
resSparseMatMul2 = tf.sparse.sparse_dense_matmul(sparse_x, y_t)


# start_time = time.time()
# options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# resStandardMatMul.eval()
# duration = time.time() - start_time
# print('Standard matmul: {}'.format(duration))

# start_time = time.time()
# resSparseMatMul0.eval()
# duration = time.time() - start_time
# print('Sparse matmul tf.transpose: {}'.format(duration))

# start_time = time.time()
# resSparseMatMul1.eval()
# duration = time.time() - start_time
# print('Sparse matmul adjoint_b=True, flag: {}'.format(duration))

# start_time = time.time()
# resSparseMatMul2.eval()
# duration = time.time() - start_time
# print('Sparse matmul without transposing: {}'.format(duration))

with tf.Session() as sess:
    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(resStandardMatMul, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('standard_matmul.json', 'w') as f:
        f.write(chrome_trace)


with tf.Session() as sess:
    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(resSparseMatMul1, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('sparse_matmul.json', 'w') as f:
        f.write(chrome_trace)

# Tensor dimensions:
#
# w[num_units + input_size][num_units * 4]
# icfo[batch_size][num_units * 4]
#
# U nas:
# num_units = 128
# batch_size = 128
# input_size = 28