import tensorflow as tf
import random
import numpy as np
from time import time
import os

LEARNING_RATE = 1e-5
BATCH_SIZE = 64
EPOCH = 10
CHECK_STEP = 20
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

data_dir = './'

class net:
    #初始化
    def __init__(self, train_file, test_file):
        os.system('rm -rf logs')
        
        #调用读取训练数据函数
        print('reading train data...')
        self.data_train = self.data_pre_train(train_file)
        #调用读取测试数据函数
        #print('reading test data...')
        #self.data_test = self.data_pre_test(test_file)
        
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_act = tf.placeholder(tf.float32, [None, 10])
        self.y_pre = self.inf(self.x)

        self.cross_entropy = tf.losses.softmax_cross_entropy(self.y_act, self.y_pre)
        tf.summary.scalar('loss', self.cross_entropy)
        #self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cross_entropy)

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    def inf(self, x):
        x_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('x_input', x_input, 6)
        with tf.name_scope('1st_layer'):
            w_conv1 = tf.get_variable('w_conv1', [3, 3, 1, 20])
            #可视化卷积核
            w_conv1_visual_0 = tf.reshape(w_conv1[:,:,:,0], [1, 3, 3, 1])
            w_conv1_visual_1 = tf.reshape(w_conv1[:,:,:,1], [1, 3, 3, 1])
            w_conv1_visual_2 = tf.reshape(w_conv1[:,:,:,2], [1, 3, 3, 1])

            tf.summary.image('w_conv1', w_conv1_visual_0)
            tf.summary.image('w_conv1', w_conv1_visual_1)
            tf.summary.image('w_conv1', w_conv1_visual_2)

            ##########
            tf.summary.histogram('w_conv1', w_conv1)
            b_conv1 = tf.get_variable('b_conv1', [20])
            tf.summary.histogram('b_conv1', b_conv1)
            h_conv1 = tf.nn.relu(self.conv2d(x_input, w_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)
            #可视化特征图
            tf.summary.image('h_pool1', h_pool1[:,:,:,:1], 6)
            ##########

        with tf.name_scope('2nd_layer'):
            w_conv2 = tf.get_variable('w_conv2', [3, 3, 20, 40])
            b_conv2 = tf.get_variable('b_conv2', [40])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2 (h_conv2)
            tf.summary.image('h_pool2', h_pool2[:,:,:,:1], 6)

        with tf.name_scope('fc_layers'):
            w_fc1 = tf.get_variable('w_fc1', [7 * 7 * 40, 1024])
            b_fc1 = tf.get_variable('b_fc1', [1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
            
        with tf.name_scope('output_layers'):
            w_fc2 = tf.get_variable('w_fc2', [1024, 10])
            b_fc2 = tf.get_variable('b_fc2', [10])
            y_pre = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2, name = 'out')

        return y_pre
    
    def data_pre_train(self, file):
        with tf.gfile.Open(data_dir + file, 'rb') as rf:
            bits = rf.read()
        s = str(bits)[2:-5]
        data = s.split('\\r\\n')
        data = data[1:]
        print(len(data))
        #print(data[0])
        #根据逗号划分每一行，去除最后的换行符
        data = [l.split(',') for l in data]
        #对划分的每个元素转换成整数
        data = [list(map(int, l)) for l in data]
        data = np.array(data)
        #新建一个每行794的矩阵
        res = np.zeros((len(data), 794))
        #后784个数字存放所有像素值，并归一化为0-1
        res[:, 10:] = np.multiply(data[:, 1:], 1.0 / 255.0)
        #前10个数字存放是否为0-9，是为1，不是为0
        for i in range(len(res)):
            res[i][data[i][0]] = 1
        print('train data shape: ', res.shape)
        return res
    
    def data_pre_test(self, file):
        with tf.gfile.Open(data_dir + file, 'rb') as rf:
            bits = rf.read()
        s = str(bits)[2:-5]
        data = s.split('\\r\\n')
        data = data[1:]
        #根据逗号划分每一行，去除最后的换行符
        data = [l.split(',') for l in data]
        #对划分的每个元素转换成整数
        data = [list(map(int, l)) for l in data]
        res = np.array(data)
        #归一化为0-1
        res = np.multiply(res, 1.0 / 255.0)
        print('test data shape: ', res.shape)
        return res

    def train(self, data, sess):
        time_start = time()

        pre = tf.argmax(self.y_pre, 1)
        act = tf.argmax(self.y_act, 1)
        correct_rate = tf.equal(tf.argmax(self.y_pre, 1), tf.argmax(self.y_act, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_rate, 'float'))
        tf.summary.scalar('accuracy', accuracy)
        
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', sess.graph, flush_secs = 10)
        
        num = len(data)
        x_valid = data[int(0.8 * num):, 10:]
        y_valid = data[int(0.8 * num):, :10]
        data = data[:int(0.8 * num)]

        i = 1
        for epoch in range(EPOCH):
            np.random.shuffle(data)
            num = len(data)
            x_train = data[:int(0.8 * num), 10:]
            #x_train = np.multiply(x_train, 1.0 / 255.0)
            y_train = data[:int(0.8 * num), :10]
            x_test = data[int(0.8 * num):, 10:]
            #x_test = np.multiply(x_test, 1.0 / 255.0)
            y_test = data[int(0.8 * num):, :10]

            data_len = len(x_train)

            start = 0
            end = BATCH_SIZE

            print('start epoch ', epoch + 1, ' / ', EPOCH)
            while(end <= data_len):

                loss = sess.run([self.train_step, self.cross_entropy, accuracy, pre, act], \
                    feed_dict = {self.x: x_train[start: end], self.y_act: y_train[start: end]})
                if i % CHECK_STEP == 0:
                    print('epoch: ', epoch + 1, 'step: ', i, 'loss: ', loss[1], 'accuracy: ', loss[2])
                    #tensorboard记录loss变化
                    result = sess.run(merged, feed_dict = {self.x: x_valid, self.y_act: y_valid})
                    writer.add_summary(result, i)
                end += BATCH_SIZE
                start += BATCH_SIZE
                i += 1

            test_acc = sess.run(accuracy, feed_dict = {self.x: x_test, self.y_act: y_test})
            valid_acc = sess.run(accuracy, feed_dict = {self.x: x_valid, self.y_act: y_valid})
            
            print('epoch: ', epoch + 1, 'test_acc: ', test_acc, 'valid_acc: ', valid_acc)
            print('************************************************************************')
            time_stop = time()
            elapsed = str(time_stop - time_start)
            print('time consumed: ', elapsed, 's')

    def test(self, data, sess):
        #res_file = open('result.csv', 'w')
        line = 'ImageId,Label'
        #res_file.writelines(line)
        #print(line)

        for i in range(int(len(data) / 1000)):
            _, y_res = sess.run([self.train_step, self.y_pre], feed_dict = {self.x: data[i * 1000 :i * 1000 + 1000], self.y_act: np.zeros((1000, 10))})
            res = sess.run(tf.argmax(y_res, 1))
            '''
            for j in range(1000):
                line = str(i * 1000 + j + 1) + ',' + str(res[j])
                #res_file.writelines(line)
                print(line)
            '''
        #res_file.close()


a = net(TRAIN_FILE, TEST_FILE)
with tf.Session() as sess:
    
    a.train(a.data_train, sess)
    #a.test(a.data_test, sess)
    
    #graph = convert_variables_to_constants(sess, sess.graph_def, ["out"])
    #tf.train.write_graph(graph, '.', 'graph.pb', as_text=False)  
    #saver = tf.train.Saver()
    #saver.save(sess, 'model.ckpt')
#a.test(a.data_test, sess)
