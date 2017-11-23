from Tools import Tools
import numpy as np
import argparse
import tensorflow as tf


class Runner:

    def __init__(self, data, model_path="model"):
        self.data = data
        self.batch_size = self.data.batch_size
        self.class_number = self.data.class_number
        self.model_path = model_path

        # 网络
        self.net = None

        self.supervisor = tf.train.Supervisor(graph=self.net.graph, logdir=self.model_path)
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    # 训练
    def train(self, epochs=10, test_freq=1, save_freq=2):
        with self.supervisor.managed_session(config=self.config) as sess:
            for epoch in range(epochs):
                # stop
                if self.supervisor.should_stop():
                    break
                # train
                for step in range(self.data.number_train):
                    x, label = self.data.next_train_batch()
                    _ = sess.run(self.net.train_op, feed_dict={self.net.x: x, self.net.label: label})
                # test
                if epoch % test_freq == 0:
                    self._test(sess, epoch)
                # save
                if epoch % save_freq == 0:
                    self.supervisor.saver.save(sess, os.path.join(self.model_path, "model_epoch_{}".format(epoch)))
            pass
        pass

    # 测试
    def test(self, info="test"):
        with self.supervisor.managed_session(config=self.config) as sess:
            self._test(sess, info)
        pass

    def _test(self, sess, info):
        test_acc = 0
        for i in range(self.data.number_test):
            x, label = self.data.next_test_batch(i)
            prediction = sess.run(self.net.prediction, {self.net.x: x})
            test_acc += np.sum(np.equal(label, prediction))
        test_acc = test_acc / (self.batch_size * self.data.number_test)
        Tools.print_info("{} {}".format(info, test_acc))
        return test_acc

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="template-Supervisor", help="name")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-class_number", type=int, default=10, help="type number")
    parser.add_argument("-data_path", type=str, default="../data/mnist", help="image data")
    args = parser.parse_args()
