import os
import tensorflow as tf

from Tools import Tools
from Data import MNIST, SVHN, CCF3, NWPU3, PreProcessing
from Net import LeNet
from Adversary import Adversary


class Runner:

    def __init__(self, source_data, target_data, net, epoch_number, batch_size, display_freq,
                 save_model_freq, save_model_path, learning_rate, weights_path, adversary_layers):
        self.source_data = source_data
        self.target_data = target_data
        self.net = net

        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.display_freq = display_freq
        self.save_model_freq = save_model_freq
        self.save_model_path = save_model_path
        self.learning_rate = learning_rate
        self.weights_path = weights_path
        self.adversary_layers = adversary_layers

        # 源域数据和标签
        self.s_x = tf.placeholder(shape=[None, self.source_data.image_size, self.source_data.image_size,
                                         self.source_data.num_channel], dtype=tf.float32)
        self.s_labels = tf.placeholder(shape=[None], dtype=tf.int32)

        # 目标域数据和标签
        self.t_x = tf.placeholder(shape=[None, self.source_data.image_size, self.source_data.image_size,
                                         self.source_data.num_channel], dtype=tf.float32)
        self.t_labels = tf.placeholder(shape=[None], dtype=tf.int32)

        # 两个交叉熵
        self.mapping_loss, self.adversary_loss = None, None
        # 三个网络的可训练变量
        self.source_vars, self.target_vars, self.adversary_vars = None, None, None
        # 优化
        self.train_mapping_op, self.train_adversary_op = self._merge_net()

        # session
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=self.config)

        pass

    # 合并三个网络（源域分类网、目标域分类网、对抗网）为一个
    def _merge_net(self):
        # 加载两个网络
        s_logits = self.net.net(self.s_x, scope="source")
        t_logits = self.net.net(self.t_x, scope="target")
        # 合并两个网络的输出
        classfies_logits = tf.concat([s_logits, t_logits], 0)  # [2 * bz, 10]

        # 所属数据域标签
        s_adversary_label = tf.zeros([tf.shape(s_logits)[0]], tf.int32)  # [bz] : all 0
        t_adversary_label = tf.ones([tf.shape(t_logits)[0]], tf.int32)  # [bz] : all 1
        # 合并所属数据域标签
        adversary_label = tf.concat([s_adversary_label, t_adversary_label], 0)  # [2 * bz]

        # 判别属于哪个数据域
        adversary_logits = Adversary.adversarial_discriminator(classfies_logits, self.adversary_layers, scope='adversary', leaky=False)

        # 两个交叉熵
        self.mapping_loss = tf.losses.sparse_softmax_cross_entropy(1 - adversary_label, adversary_logits)  # source loss
        self.adversary_loss = tf.losses.sparse_softmax_cross_entropy(adversary_label, adversary_logits)  # target loss

        # 得到三个网络的可训练变量
        self.source_vars = Tools.collect_vars('source')
        self.target_vars = Tools.collect_vars('target')
        self.adversary_vars = Tools.collect_vars('adversary')

        # 优化
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        train_mapping_op = optimizer.minimize(self.mapping_loss, var_list=list(self.target_vars.values()))
        train_adversary_op = optimizer.minimize(self.adversary_loss, var_list=list(self.adversary_vars.values()))
        return train_mapping_op, train_adversary_op

    # 加载并初始化网络的权值
    def _init_weights(self):
        # 加载可训练变量
        if os.path.isdir(self.weights_path):
            weights = tf.train.latest_checkpoint(self.weights_path)
        else:
            raise FileNotFoundError()
        # 初始化源域的可训练变量
        s_restorer = tf.train.Saver(var_list=self.source_vars)
        s_restorer.restore(self.sess, weights)
        # 初始化目标域的可训练变量
        t_restorer = tf.train.Saver(var_list=self.target_vars)
        t_restorer.restore(self.sess, weights)
        return s_restorer, t_restorer

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        # 加载并初始化网络的权值
        s_restorer, t_restorer = self._init_weights()

        for epoch in range(self.epoch_number):
            # train
            total_mapping_loss = 0.0
            total_adversary_loss = 0.0
            for step in range(self.source_data.number_train):
                # input
                s_x, s_labels = self.source_data.next_train_batch()
                t_x, t_labels = self.target_data.next_train_batch()
                # 处理数据：将源域数据和目标域数据处理成一样
                s_x = PreProcessing()(s_x, self.net, one_image=False)
                t_x = PreProcessing()(t_x, self.net, one_image=False)
                # run get loss
                mapping_loss, adversary_loss, _, _ = self.sess.run(
                    [self.mapping_loss, self.adversary_loss,self.train_mapping_op, self.train_adversary_op],
                    feed_dict={self.s_x: s_x, self.s_labels: s_labels,self.t_x: t_x, self.t_labels: t_labels})
                # print loss
                if step % self.display_freq == 0:
                    Tools.print_info("{}/{} mapping loss is {}, adversary loss is {}".format(
                        step, self.source_data.number_train, mapping_loss, adversary_loss))
                    pass
                # stat loss
                total_mapping_loss += mapping_loss
                total_adversary_loss += adversary_loss
            # print avg loss
            Tools.print_info("avg mapping loss is {}, adversary loss is {}".format(
                total_mapping_loss / self.source_data.number_train, total_adversary_loss / self.source_data.number_train))

            # save
            if epoch % self.save_model_freq == 0:
                t_restorer.save(self.sess, os.path.join(self.save_model_path, "model_epoch_{}".format(epoch)))
            Tools.print_info("epoch {} over".format(epoch))
            pass
        pass

    pass


def run_mnist_to_svhn():
    batch_size = 64
    s_data = MNIST(batch_size, num_classes=10, data_path="data/mnist")
    t_data = SVHN(batch_size, num_classes=10, data_path="data/svhn")
    net = LeNet(image_size=s_data.image_size, num_channel=s_data.num_channel)

    runner = Runner(source_data=s_data, target_data=t_data, net=net,
                    epoch_number=10, batch_size=batch_size, display_freq=100, save_model_freq=2,
                    save_model_path=Tools.new_dir("model/lenet_adda/mnist_to_svhn"), learning_rate=1e-4,
                    weights_path="model/lenet/mnist", adversary_layers=[500, 500])
    runner.train()
    pass


def run_svhn_to_mnist():
    batch_size = 64
    s_data = SVHN(batch_size, num_classes=10, data_path="data/svhn")
    t_data = MNIST(batch_size, num_classes=10, data_path="data/mnist")
    net = LeNet(image_size=s_data.image_size, num_channel=s_data.num_channel)

    runner = Runner(source_data=s_data, target_data=t_data, net=net,
                    epoch_number=10, batch_size=batch_size, display_freq=100, save_model_freq=1,
                    save_model_path=Tools.new_dir("model/lenet_adda/svhn_to_mnist"), learning_rate=1e-4,
                    weights_path="model/lenet/svhn", adversary_layers=[500, 500])
    runner.train()
    pass


def run_ccf3_to_nwpu3():
    batch_size = 64
    s_data = CCF3(batch_size, num_classes=4, data_path="data/ccf3")
    t_data = NWPU3(batch_size, num_classes=4, data_path="data/nwpu3")
    net = LeNet(image_size=s_data.image_size, num_channel=s_data.num_channel)

    runner = Runner(source_data=s_data, target_data=t_data, net=net,
                    epoch_number=20, batch_size=batch_size, display_freq=10, save_model_freq=1,
                    save_model_path=Tools.new_dir("model/lenet_adda/ccf3_to_nwpu3"), learning_rate=1e-4,
                    weights_path="model/lenet/ccf3", adversary_layers=[500, 500])
    runner.train()
    pass


def run_nwpu3_to_ccf3():
    batch_size = 64
    s_data = NWPU3(batch_size, num_classes=3, data_path="data/nwpu3")
    t_data = CCF3(batch_size, num_classes=3, data_path="data/ccf3")
    net = LeNet(image_size=s_data.image_size, num_channel=s_data.num_channel)

    runner = Runner(source_data=s_data, target_data=t_data, net=net,
                    epoch_number=3, batch_size=batch_size, display_freq=10, save_model_freq=1,
                    save_model_path=Tools.new_dir("model/lenet_adda/nwpu3_to_ccf3"), learning_rate=1e-4,
                    weights_path="model/lenet/nwpu3", adversary_layers=[500, 500])
    runner.train()
    pass

if __name__ == '__main__':
    # run_mnist_to_svhn()
    # run_svhn_to_mnist()
    run_ccf3_to_nwpu3()
    pass
