import numpy as np
import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('variables'):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='total_output')

    with tf.name_scope('transformation'):
        # 输入层
        with tf.name_scope('input'):
            a = tf.placeholder(dtype=tf.float32, shape=None, name='input_a')

        # 中间层
        with tf.name_scope('intermediate_layer'):
            b = tf.reduce_prod(a, name='product_b')
            c = tf.reduce_sum(a, name='sum_c')

        # 输出层
        with tf.name_scope('output'):
            output = tf.add(b, c, name='output')

    with tf.name_scope('update'):
        update_total = total_output.assign_add(output)
        increment_step = global_step.assign_add(1)

    with tf.name_scope('summaries'):
        avg = tf.divide(update_total, tf.cast(increment_step, tf.float32), name='average')

        # 为输出节点创建汇总数据
        tf.summary.scalar('output', output)
        tf.summary.scalar('Sum_of_output', update_total)
        tf.summary.scalar('Average_of_output', avg)

    with tf.name_scope('global_ops'):
        # 初始化
        init = tf.global_variables_initializer()
        merged_summaries = tf.summary.merge_all()

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./improved_graph', graph=graph)
sess.run(init)


def run_graph(input_tensor):
    fed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=fed_dict)
    writer.add_summary(summary, global_step=step)


run_graph([2, 8])
run_graph([3, 1, 3, 3])
run_graph([8])
run_graph([1, 2, 3])
run_graph([11, 4])
run_graph([4, 1])
run_graph([7, 3, 1])
run_graph([6, 3])
run_graph([0, 2])
run_graph([4, 5, 6])

# 将汇总数据写入磁盘
writer.flush()
# 清理工作
writer.close()
sess.close()
