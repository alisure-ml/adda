import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tensorflow.examples.tutorials.mnist import input_data


class PreProcessing:

    def __call__(self, inputs, net, one_image):
        channels = len(inputs[0][0][0])
        if channels == 1 and net.num_channel == 3:
            inputs = self.gray2rgb(inputs) if one_image else [self.gray2rgb(input_one) for input_one in inputs]
        elif channels == 3 and net.num_channel == 1:
            inputs = self.rgb2gray(inputs) if one_image else [self.rgb2gray(input_one) for input_one in inputs]
        # 要求图片是方的
        if net.image_size is not None and net.image_size != len(inputs[0]):
                if one_image:
                    inputs = self.resize(inputs, net.image_size, net.num_channel)
                else:
                    inputs = [self.resize(input_one, net.image_size, net.num_channel) for input_one in inputs]
        return inputs

    # TODO: 该函数需要优化
    @staticmethod
    def resize(image, image_size, num_channel):
        image = np.asarray(image * 255, dtype=np.uint8)
        if num_channel == 1:
            image = np.squeeze(image)
            image = np.asarray(Image.fromarray(image).convert("L").resize((image_size, image_size))) / 255.0
            image = np.reshape(image, newshape=[len(image), len(image[0]), 1])
        else:
            image = np.asarray(Image.fromarray(image).convert("RGB").resize((image_size, image_size))) / 255.0
        return image

    @staticmethod
    def resize_images(images, image_size, num_channel):
        return np.asarray([PreProcessing.resize(image, image_size, num_channel) for image in images])

    @staticmethod
    def rgb2gray(image):
        RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        return np.sum(np.multiply(image, RGB2GRAY), axis=2, keepdims=True)

    @staticmethod
    def gray2rgb(image):
        RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        return np.multiply(image, RGB2GRAY)

    pass


class MNIST:

    def __init__(self, batch_size, num_classes, data_path):
        self.batch_size = batch_size
        self.num_classes = num_classes

        self._mnist = input_data.read_data_sets(data_path, reshape=False)
        self._data_train = self._mnist.train
        self._data_test = self._mnist.test

        self.number_train = self._data_train.num_examples // self.batch_size
        self.number_test = self._data_test.num_examples // self.batch_size

        self.num_channel = len(self._data_train.images[0][0][0])
        self.image_size = len(self._data_train.images[0])
        pass

    def next_train_batch(self):
        return self._data_train.next_batch(self.batch_size)

    def next_test_batch(self, index):
        start = 0 if index >= self.number_test else index * self.batch_size
        end = self.batch_size if index >= self.number_test else (index + 1) * self.batch_size
        return self._data_test.images[start: end], self._data_test.labels[start: end]

    pass


class SVHN:

    def __init__(self, batch_size, num_classes, data_path):
        self.batch_size = batch_size
        self.num_classes = num_classes

        # 数据文件
        self.data_files = {'train': os.path.join(data_path, 'train_32x32.mat'),
                           'test': os.path.join(data_path, 'test_32x32.mat')}

        # 数据和标签
        self.train_images, self.train_labels, self.test_images, self.test_labels = self._load_data()

        # 一些统计信息
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len(self.train_images)
        self.number_train = self._num_examples // self.batch_size
        self.number_test = len(self.test_images) // self.batch_size

        # 通道数和图片大小
        self.num_channel = len(self.train_images[0][0][0])  # 3
        self.image_size = len(self.train_images[0])  # 32
        pass

    def _load_data(self):
        train_mat = loadmat(self.data_files["train"])
        train_images = train_mat['X'].transpose((3, 0, 1, 2)).astype(np.float32) / 255
        train_labels = train_mat['y'].squeeze()
        train_labels[train_labels == self.num_classes] = 0

        test_mat = loadmat(self.data_files['test'])
        test_images = test_mat['X'].transpose((3, 0, 1, 2)).astype(np.float32) / 255
        test_labels = test_mat['y'].squeeze()
        test_labels[test_labels == self.num_classes] = 0
        return train_images, train_labels, test_images, test_labels

    def next_train_batch(self):
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size
        if self._index_in_epoch > len(self.train_images):
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size
            assert self.batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.train_images[start:end], self.train_labels[start:end]

    def next_test_batch(self, index):
        start = 0 if index >= self.number_test else index * self.batch_size
        end = self.batch_size if index >= self.number_test else (index + 1) * self.batch_size
        return self.test_images[start: end], self.test_labels[start: end]

    pass


class CCF3:

    def __init__(self, batch_size, num_classes, data_path):
        self.batch_size = batch_size
        self.num_classes = num_classes

        # 数据文件
        self.data_files = {'train': os.path.join(data_path, 'train_256x256.mat'),
                           'test': os.path.join(data_path, 'test_256x256.mat')}

        # 数据和标签
        self.train_images, self.train_labels, self.test_images, self.test_labels = self._load_data()
        self.train_images = PreProcessing.resize_images(self.train_images, 32, 3)
        self.test_images = PreProcessing.resize_images(self.test_images, 32, 3)

        # 一些统计信息
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len(self.train_images)
        self.number_train = self._num_examples // self.batch_size
        self.number_test = len(self.test_images) // self.batch_size

        # 通道数和图片大小
        self.num_channel = len(self.train_images[0][0][0])  # 3
        self.image_size = len(self.train_images[0])  # 32
        pass

    def _load_data(self):
        train_mat = loadmat(self.data_files["train"])
        train_images = train_mat['x'].transpose((3, 0, 1, 2)).astype(np.float32) / 255
        train_labels = np.asarray(train_mat['y'].squeeze(), dtype=np.int32)
        train_labels[train_labels == self.num_classes] = 0

        test_mat = loadmat(self.data_files['test'])
        test_images = test_mat['x'].transpose((3, 0, 1, 2)).astype(np.float32) / 255
        test_labels = np.asarray(test_mat['y'].squeeze(), dtype=np.int32)
        test_labels[test_labels == self.num_classes] = 0
        return train_images, train_labels, test_images, test_labels

    def next_train_batch(self):
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size
        if self._index_in_epoch > len(self.train_images):
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size
            assert self.batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.train_images[start:end], self.train_labels[start:end]

    def next_test_batch(self, index):
        start = 0 if index >= self.number_test else index * self.batch_size
        end = self.batch_size if index >= self.number_test else (index + 1) * self.batch_size
        return self.test_images[start: end], self.test_labels[start: end]

    pass


class NWPU3:

    def __init__(self, batch_size, num_classes, data_path):
        self.batch_size = batch_size
        self.num_classes = num_classes

        # 数据文件
        self.data_files = {'train': os.path.join(data_path, 'train_256x256.mat'),
                           'test': os.path.join(data_path, 'test_256x256.mat')}

        # 数据和标签
        self.train_images, self.train_labels, self.test_images, self.test_labels = self._load_data()
        self.train_images = PreProcessing.resize_images(self.train_images, 32, 3)
        self.test_images = PreProcessing.resize_images(self.test_images, 32, 3)

        # 一些统计信息
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len(self.train_images)
        self.number_train = self._num_examples // self.batch_size
        self.number_test = len(self.test_images) // self.batch_size

        # 通道数和图片大小
        self.num_channel = len(self.train_images[0][0][0])  # 3
        self.image_size = len(self.train_images[0])  # 32
        pass

    def _load_data(self):
        train_mat = loadmat(self.data_files["train"])
        train_images = train_mat['x'].transpose((3, 0, 1, 2)).astype(np.float32) / 255
        train_labels = np.asarray(train_mat['y'].squeeze(), dtype=np.int32)
        train_labels[train_labels == self.num_classes] = 0

        test_mat = loadmat(self.data_files['test'])
        test_images = test_mat['x'].transpose((3, 0, 1, 2)).astype(np.float32) / 255
        test_labels = np.asarray(test_mat['y'].squeeze(), dtype=np.int32)
        test_labels[test_labels == self.num_classes] = 0
        return train_images, train_labels, test_images, test_labels

    def next_train_batch(self):
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size
        if self._index_in_epoch > len(self.train_images):
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size
            assert self.batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.train_images[start:end], self.train_labels[start:end]

    def next_test_batch(self, index):
        start = 0 if index >= self.number_test else index * self.batch_size
        end = self.batch_size if index >= self.number_test else (index + 1) * self.batch_size
        return self.test_images[start: end], self.test_labels[start: end]

    pass


if __name__ == '__main__':
    # MNIST(10, 10, "data/mnist")
    # SVHN(10, 10, "data/svhn")
    CCF3(10, 3, "data/ccf3")
