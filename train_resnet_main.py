from tensorflow.examples.tutorials.mnist import input_data
import collections
import tensorflow as tf
from datetime import datetime
import math
import time
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

slim = tf.contrib.slim

'''
使用collectin.namedtuple设计ResNet基本Block模块组
示例：
    MyTupleClass = collections.namedtuple('MyTupleClass',['name', 'age', 'job'])
    obj = MyTupleClass("Tomsom",12,'Cooker')
    print(obj.name)
    print(obj.age)
    print(obj.job)
执行结果：
    Tomsom
    12
    Cooker
'''
Block = collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])

def subsample(inputs, factor, scope = None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride = factor, scope = scope)

# 依据步长选择卷积策略
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:  # 如果stride == 1, 直接进行卷积
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = 1,
                           padding = 'SAME', scope = scope)
    else:     # 如果stride ！= 1
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2     # 上方、左方填充0的列数（行数）
        pad_end = pad_total - pad_beg    # 下方、右方填充0的列数（行数）
        inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_end], [pad_beg, pad_end], [0,0]]) # 进行全0填充
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = stride, padding = 'VALID', scope = scope)
        
# 下面等价于slim.add_arg_scope(stack_blocks_dense)
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections = None):
    '''
    定义堆叠Blocks的函数
    net: 输入
    blocks: 之前定义的Block的class的列表
    outputs_collections: 用来收集各个end_point的collections
    '''
    for block in blocks:
        with tf.compat.v1.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.compat.v1.variable_scope('unit_%d' % (i+1), values = [net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    # unit_fn: 残差学习单元的生成函数，顺序地创建并连接所有的残差学习单元
                    net = block.unit_fn(net, depth = unit_depth, 
                                        depth_bottleneck = unit_depth_bottleneck,
                                        stride = unit_stride)
                # collect_named_outputs: 将输出net添加到collection中
                net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


# 创立ResNet通用的arg_scope
def resnet_arg_scope(is_training = True,
                     weight_decay = 0.0001,
                     batch_norm_decay = 0.997,
                     batch_norm_epsilon = 1e-5,
                     batch_norm_scale = True):
    '''
    weight_decay: 权重衰减速率, 即下面L2所占比
    batch_norm_decay: BN衰减速率
    batch_norm_epsilon： BN的epsilon
    batch_norm_scale: BN的scale默认为True, 即乘以公式中的gamma
    '''
    batch_norm_paras = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.compat.v1.GraphKeys.UPDATE_OPS
    }
    # 设置slim.conv2d()中的参数默认值
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer = slim.l2_regularizer(weight_decay),
                        weights_initializer = slim.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu,
                        normalizer_fn = slim.batch_norm,
                        normalizer_params = batch_norm_paras):
        # 设置slim.batch_norm中的参数默认值，**batch_norm_para是解包的作用，把对应参数值分配给BN中的参数
        with slim.arg_scope([slim.batch_norm], **batch_norm_paras):
            # 设置最大池化的默认参数
            with slim.arg_scope([slim.max_pool2d], padding = 'SAME') as arg_sc:
                return arg_sc


# 定义核心的bottleneck残差学习单元(是ResNet V2论文提到的Full Preactivation Residual Net的一个变种)
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections = None, scope = None):
    '''
    inputs: 输入
    depth, depth_bottleneck, stride: Blocks类中的args
    outputs_collection: 收集end_points的collection
    scope: 当前unit的名称
    '''
    with tf.compat.v1.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # 获取输入的最后一个维度
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank = 4)
        # 对输入进行预BN操作
        preact = slim.batch_norm(inputs, activation_fn = tf.nn.relu, scope = 'preact')

        if depth == depth_in: # 如果残差单元的输入通道数depth_in与输出通道数depth相同
            shortcut = subsample(inputs, stride, 'shortcut')
        else:            # 如果输入与输出通道不一致，则用stride=1的卷积操作改变通道数
            shortcut = slim.conv2d(preact, depth, [1, 1], stride = stride,
                                   normalizer_fn = None, 
                                   activation_fn = None,
                                   scope = 'shortcut')
        # 输出通道数为depth_bottleneck的卷积, 卷积核1x1
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1],
                               stride = 1, scope = 'conv1')
        # 3表示卷积核尺寸3x3(帮你看代码抗"过拟合")
        residual = conv2d_same(residual, depth_bottleneck, 3,
                                    stride, scope = 'conv2')
        # 输出通道数为depth, 卷积核1x1
        residual = slim.conv2d(residual, depth, [1, 1], stride = 1,
                               normalizer_fn = None, activation_fn = None,
                               scope = 'conv3')
        # 实现Residual output的结果
        output = residual + shortcut
        # 将结果添加入collection，并返回output作为函数结果
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)



def resnet_v2(inputs, blocks, num_classes = None,
              global_pool = True, 
              include_root_block = True,
              reuse = None,
              scope = None):
    '''
    inputs: 输入
    blocks: 定义好的Block类的列表
    num_classes: 最后输出的类数
    global_pool: 标志是否加上最后一层的全局平均池化
    include_root_block: 标志是否加上ResNet网络最前面通常使用的7x7卷积和最大池化
    reuse: 标志是否重用
    scope: 整个网络的名称
    '''
    with tf.compat.v1.variable_scope(scope, 'resnet_v2', [inputs], reuse = reuse) as sc:
        end_points_collection =sc.original_name_scope + '_end_point'
        # 设置outputs_collections默认参数为end_points_collection
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections = end_points_collection):
            net = inputs
            if include_root_block:
                # 设置slim.conv2d的默认参数
                with slim.arg_scope([slim.conv2d], activation_fn = None,
                                    normalizer_fn = None):
                    # 创建ResNet最前面64输出通道步长为2的7x7卷积
                    net = conv2d_same(net, 64, 7, stride = 2, scope = 'conv1')
                # 接步长为2的3x3池化
                net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'pool1')   # 执行完后，图片尺寸以缩小为1/4
            # 用stack_blocks_dense把残差学习模块生成好
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn = tf.nn.relu, scope = 'postnorm')
            if global_pool: # 如果要进行全局平均池化层
                net = tf.reduce_mean(net, [1, 2], name = 'pool5', keep_dims = True)
            if num_classes is not None:  # 用卷积操作替代全连接层(添加一个输出通道为num_classes的1x1卷积)
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'logits')
                net = slim.flatten(net)
            # 将collection转为dict
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            '''
            if num_classes is not None:
                # net = tf.reshape(net, [-1, num_classes])
                # net = tf.squeeze(input=net, axis=[1,2])
                # end_points['predictions'] = tf.squeeze(input=net, axis=[1,2])
                # end_points['predictions'] = slim.softmax(net, scope = 'prediction')
                end_points['predictions'] = tf.nn.softmax(net, name='predictions')
            ''' 

            return net, end_points


# 50层深度的ResNet网络配置
def resnet_v2_50(inputs, num_classes = None,
                 global_pool = True,
                 reuse = None,
                 scope = 'resnet_v2_50'):
    '''
    以下面Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)])为例
    block1: 是这个Block的名称
    bottleneck: 前面定义的残差学习单元（有三层）
    [(256, 64, 1)] * 2 + [(256, 64, 2)]: 是一个列表，其中每个元素都对应一个bottleneck
        残差学习单元，前面两个元素都是(256, 64, 1),最后一个是(256, 64, 2)。每个元素都
        时一个3元组，即（depth, depth_bottleneck, stride）,代表构建的bottleneck残差学
        习单元中，第三层的输出通道为256（depth），前两层的输出通道数为64（depth_bottleneck）
        且中间那层的步长stride为1（stride）
    '''
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block = False, reuse = reuse,
                     scope = scope)

# 101层深度的ResNet网络配置
def resnet_v2_101(inputs, num_classes = None,
                  global_pool = True,
                  reuse = None,
                  scope = 'resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2+ [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block = True, reuse = reuse,
                     scope = scope)

# 152层深度的ResNet网络配置
def resnet_v2_152(inputs, num_classes = None,
                  global_pool = True,
                  reuse = None,
                  scope = 'resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block = True, reuse = reuse,
                     scope = scope)

# 200层深度的ResNet网络配置
def resnet_v2_200(inputs, num_classes = None,
                  global_pool = True,
                  reuse = None,
                  scope = 'resnet_v2_200'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block = True, reuse = reuse,
                     scope = scope)






def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10  # 打印阈值
    total_duration = 0.0    # 每一轮所需要的迭代时间
    total_duration_aquared = 0.0  # 每一轮所需要的迭代时间的平方
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time    # 计算耗时
        if i >= num_steps_burn_in:
            if not i % 10:
                # print("%s : step %d, duration = %.3f" % (datetime.now(), i - num_steps_burn_in, duration))
                print('{} : step {}, duration={}'.format(datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_aquared += duration * duration
    mn = total_duration / num_batches   # 计算均值
    vr = total_duration_aquared / num_batches - mn * mn  # 计算方差
    sd = math.sqrt(vr) # 计算标准差
    # print("%s : %s across %d steps, %.3f +/- %.3f sec/batch" % (datetime.now(), info_string, num_batches, mn, sd))
    print('{}:{} across {} steps, {} +/- {} sec/batch'.format(datetime.now(), info_string, num_batches, mn, sd))



batch_size = 50
height, width = 28, 28
learning_rate_base = 0.01
regulariztion_rate = 0.0001
learing_rate_decay = 0.99
training_steps = 5000
moving_average_decay = 0.99
image_size = 28
num_channels = 1
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 占位符
# 28X28=784  第一个None 为batch size
x = tf.compat.v1.placeholder("float", shape = [None, height, width, 1], name='x-input')
# 输出10个类别
y_ = tf.compat.v1.placeholder("float", shape = [None, 10], name='y-output')
# x_image = tf.reshape(x, [-1, 28, 28, 1])





train_output = './checkpoint'
if not tf.io.gfile.exists(train_output):
  tf.gfile.MakeDirs(train_output)

train_log_dir = './logs'
if not tf.io.gfile.exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)


y, end_points = resnet_v2_50(inputs=x, num_classes=10)
global_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())

# predictions = net(images, is_training=True)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
cross_entropy = tf.reduce_mean(cross_entropy)

tf.compat.v1.add_to_collection('loss' , cross_entropy)

regularizer_loss = tf.add_n(tf.compat.v1.get_collection('loss'))

learning_rate = tf.compat.v1.train.exponential_decay(
                learning_rate_base,
                global_step,
                mnist.train.num_examples / batch_size,
                learing_rate_decay,
                staircase=True)

train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(regularizer_loss, global_step=global_step)

with tf.control_dependencies([train_step , variables_averages_op]):
    train_op = tf.no_op(name='train')
    
correct_prediction = tf.equal(tf.argmax(y , 1) ,tf.argmax(y_ , 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))


saver=tf.compat.v1.train.Saver(max_to_keep=5)

with tf.compat.v1.Session() as sess :
    tf.compat.v1.global_variables_initializer().run()
    train_writer = tf.compat.v1.summary.FileWriter(r'logs/', sess.graph)
    #==============================================================================
    validate_feed  = {x : np.reshape(mnist.validation.images[:200],(-1, image_size, image_size, num_channels)),y_:mnist.validation.labels[:200]}
    test_feed = {x:np.reshape(mnist.test.images[:200],(-1 ,image_size, image_size, num_channels)) , y_:mnist.test.labels[:200]}
    every_tranin = int(mnist.train.num_examples / batch_size ) 
    for i in range(training_steps):
        for j in range(every_tranin):
            bx , by = mnist.train.next_batch(batch_size)
            _ ,  step = sess.run([train_op  , global_step] , feed_dict={x:np.reshape(bx,(-1 ,image_size, image_size, num_channels)) , y_:by})
        if i % 2 == 0 :
            validate_acc = sess.run(accuracy , feed_dict=validate_feed)
            print("After %d training step(s), global_step is (%s) ,validation accuracy using average model is %g " % (i, step, validate_acc))
            saver.save(sess , 'saver/moedl1.ckpt',global_step=global_step)
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(training_steps, test_acc)))
             
    #=============================================================================






