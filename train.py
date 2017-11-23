import argparse
import os
import tensorflow as tf

from Tools import Tools
from Data import MNIST, SVHN, PreProcessing
from Net import LeNet


class Runner:

    def __init__(self, data, net, epoch_number, batch_size, display_freq, save_model_freq, save_model_path, learning_rate):
        self.data = data
        self.net = net
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.display_freq = display_freq
        self.save_model_freq = save_model_freq
        self.save_model_path = save_model_path
        self.learning_rate = learning_rate

        # 输入
        self.x = tf.placeholder(shape=[None, self.data.image_size, self.data.image_size, self.data.num_channel],
                                dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32)

        # 网络输出
        self.logits = self.net.net(self.x, scope=self.net.default_scope)

        # 损失和训练节点
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # session
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=self.config)
        pass

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        # 收集变量
        model_vars = Tools.collect_vars(scope=self.net.default_scope)
        saver = tf.train.Saver(var_list=model_vars)

        for epoch in range(self.epoch_number):
            # train
            total_loss = 0.0
            for step in range(self.data.number_train):
                x, labels = self.data.next_train_batch()
                # 处理数据：将源域数据和目标域数据处理成一样
                x = PreProcessing()(x, self.net, one_image=False)
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.x: x, self.labels: labels})
                if step % 100 == 0:
                    Tools.print_info("{}/{} loss is {}".format(step, self.data.number_train, loss))
                total_loss += loss
            Tools.print_info("avg loss is {}".format(total_loss / self.data.number_train))
            # save
            if epoch % self.save_model_freq == 0:
                saver.save(self.sess, os.path.join(self.save_model_path, "model_epoch_{}".format(epoch)))
            Tools.print_info("epoch {} over".format(epoch))
            pass
        pass

    pass


def run_mnist():
    batch_size = 128
    data = MNIST(batch_size, num_classes=10, data_path="data/mnist")
    net = LeNet(image_size=data.image_size, num_channel=data.num_channel)

    runner = Runner(data, net, epoch_number=5, batch_size=batch_size, display_freq=1, save_model_freq=1,
                    save_model_path=Tools.new_dir("model/lenet/mnist"), learning_rate=1e-4)
    runner.train()
    pass


def run_svhn():
    batch_size = 128
    data = SVHN(batch_size, num_classes=10, data_path="data/svhn")
    net = LeNet(image_size=data.image_size, num_channel=data.num_channel)

    runner = Runner(data, net, epoch_number=5, batch_size=batch_size, display_freq=1, save_model_freq=1,
                    save_model_path=Tools.new_dir("model/lenet/svhn"), learning_rate=1e-4)
    runner.train()
    pass

if __name__ == '__main__':
    # run_mnist()
    run_svhn()
    pass
