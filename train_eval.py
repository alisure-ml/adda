import os
import numpy as np
from Data import MNIST, SVHN, PreProcessing
from Net import LeNet
import tensorflow as tf
from Tools import Tools


class Runner:

    def __init__(self, data, net, s_num_channel, s_image_size, batch_size, weights_path):
        self.data = data
        self.net = net
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.s_num_channel = s_num_channel
        self.s_image_size = s_image_size

        # 输入
        self.x = tf.placeholder(shape=[None, self.s_image_size, self.s_image_size, self.s_num_channel],
                                dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32)

        # 预测
        self.logits = self.net.net(self.x, scope="target")  # [bz, 10]
        self.prediction = tf.squeeze(tf.argmax(self.logits, axis=1))

        # session
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=self.config)
        pass

    def eval(self):
        self.sess.run(tf.global_variables_initializer())

        # 收集变量，加载模型
        model_vars = Tools.collect_vars(scope="target")
        restorer = tf.train.Saver(var_list=model_vars)
        if os.path.isdir(self.weights_path):
            weights = tf.train.latest_checkpoint(self.weights_path)
        else:
            raise FileNotFoundError()
        restorer.restore(self.sess, weights)

        # train
        class_correct = np.zeros(shape=[self.data.num_classes], dtype=np.int32)
        class_counter = np.zeros(shape=[self.data.num_classes], dtype=np.int32)
        for index in range(self.data.number_test):
            # input
            x, labels = self.data.next_test_batch(index)
            # 处理数据：将源域数据和目标域数据处理成一样
            x = PreProcessing()(x, self.net, one_image=False)
            # run get prediction
            prediction = self.sess.run(self.prediction, feed_dict={self.x: x, self.labels: labels})
            # stat
            for label_index, label in enumerate(labels):
                class_counter[label] += 1
                if prediction[label_index] == label:
                    class_correct[label] += 1
            pass
        # print result
        Tools.print_info("Class accuracy:")
        Tools.print_info("  ".join(["{:.3f}".format(class_correct[i] / x) for i, x in enumerate(class_counter)]))
        Tools.print_info("Overall accuracy {}".format(np.sum(class_correct) / np.sum(class_counter)))
        pass
    pass


def run_shvn_on_mnist():
    batch_size = 128
    # 注意：需要参数
    s_image_size = 28
    s_num_channel = 1
    t_data = SVHN(batch_size, num_classes=10, data_path="data/svhn")
    net = LeNet(image_size=s_image_size, num_channel=s_num_channel)

    runner = Runner(data=t_data, net=net, s_num_channel=s_num_channel, s_image_size=s_image_size,
                    batch_size=batch_size, weights_path="model/lenet_adda/mnist_to_svhn")
    runner.eval()
    pass


def run_mnist_on_shvn():
    batch_size = 128
    # 注意：需要参数
    s_image_size = 32
    s_num_channel = 3
    t_data = MNIST(batch_size, num_classes=10, data_path="data/mnist")
    net = LeNet(image_size=s_image_size, num_channel=s_num_channel)

    runner = Runner(data=t_data, net=net, s_num_channel=s_num_channel, s_image_size=s_image_size,
                    batch_size=batch_size, weights_path="model/lenet_adda/svhn_to_mnist")
    runner.eval()
    pass


def run_mnist_on_shvn_no_a():
    batch_size = 128
    # 注意：需要参数
    s_image_size = 32
    s_num_channel = 3
    t_data = MNIST(batch_size, num_classes=10, data_path="data/mnist")
    net = LeNet(image_size=s_image_size, num_channel=s_num_channel)

    runner = Runner(data=t_data, net=net, s_num_channel=s_num_channel, s_image_size=s_image_size,
                    batch_size=batch_size, weights_path="model/lenet/svhn")
    runner.eval()
    pass


def run_mnist_on_mnist():
    batch_size = 128
    # 注意：需要参数
    s_image_size = 28
    s_num_channel = 1
    t_data = MNIST(batch_size, num_classes=10, data_path="data/mnist")
    net = LeNet(image_size=s_image_size, num_channel=s_num_channel)

    runner = Runner(data=t_data, net=net, s_num_channel=s_num_channel, s_image_size=s_image_size,
                    batch_size=batch_size, weights_path="model/lenet/mnist")
    runner.eval()
    pass


def run_shvn_on_shvn():
    batch_size = 128
    # 注意：需要参数
    s_image_size = 32
    s_num_channel = 3
    t_data = SVHN(batch_size, num_classes=10, data_path="data/svhn")
    net = LeNet(image_size=s_image_size, num_channel=s_num_channel)

    runner = Runner(data=t_data, net=net, s_num_channel=s_num_channel, s_image_size=s_image_size,
                    batch_size=batch_size, weights_path="model/lenet_adda/svhn_to_mnist")
    runner.eval()
    pass

if __name__ == '__main__':
    # run_shvn_on_mnist()
    run_mnist_on_shvn()
    # run_mnist_on_mnist()
    # run_shvn_on_shvn()
    # run_mnist_on_shvn_no_a()
    pass

