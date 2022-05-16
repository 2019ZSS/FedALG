import numpy as np
import gzip
import os
import platform
import pickle
import torch
import torchvision
# from torchvision import transforms as transforms
from torchvision import transforms


class MyDataset(torch.utils.data.Dataset):

    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.size(0)


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.num_cls = None # 数据集类别数量
        self.train_data = None # 训练集
        self.train_label = None # 标签
        self.train_data_size = None # 训练数据的大小
        self.train_data_classes = None
        self.train_transforms = None
        self.test_data = None   # 测试数据集
        self.test_label = None  # 测试的标签
        self.test_data_size = None # 测试集数据Size
        self.test_transforms = None
        
        self._index_in_train_epoch = 0
        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name == 'cifar10':
            self.load_cifar10_data()
        elif self.name == 'mnist_v2':
            self.load_mnist_data()
        elif self.name == 'emnist':
            self.load_emnist_data()
        else:
            pass
    
    def load_mnist_data(self):
        train_transforms = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.ToTensor(), 
            transforms.Normalize((0.13065973,), (0.3015038,))
            ])
        test_transforms = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.ToTensor(), 
            transforms.Normalize((0.13065973,), (0.3015038,))
            ])
        train_data = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=train_transforms)
        test_data = torchvision.datasets.MNIST(root="./data", download=True, train=False, transform=test_transforms)

        self.num_cls =  len(train_data.classes)
        self.train_data_classes = train_data.classes
        self.train_data = train_data.data
        self.train_label = train_data.targets
        self.train_data_size = train_data.data.shape[0]
        self.train_transforms = train_transforms
        self.test_data = test_data.data
        self.test_label = test_data.targets
        self.test_data_size = test_data.data.shape[0]
        self.test_transforms = test_transforms

    def load_emnist_data(self):

        train_transforms = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.ToTensor(), 
            transforms.Normalize((0.1735879,), (0.32482338,))]
        )
        test_transforms = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.ToTensor(), 
            transforms.Normalize((0.1735879,), (0.32482338,))]
        )
        try:
            train_data = torchvision.datasets.EMNIST(root="./data/", split="byclass", download=True, train=True, transform=train_transforms)
            test_data = torchvision.datasets.EMNIST(root="./data/", split="byclass", download=True, train=False, transform=test_transforms)
        except Exception as e:
            print(e)
            train_data = torchvision.datasets.EMNIST(root="./data/", split="byclass", download=False, train=True, transform=train_transforms)
            test_data = torchvision.datasets.EMNIST(root="./data/", split="byclass", download=False, train=False, transform=test_transforms)

        self.num_cls =  len(train_data.classes)
        self.train_data_classes = train_data.classes
        self.train_data = train_data.data
        self.train_label = train_data.targets
        self.train_data_size = train_data.data.shape[0]
        self.train_transforms = train_transforms
        self.test_data = test_data.data
        self.test_label = test_data.targets
        self.test_data_size = test_data.data.shape[0]
        self.test_transforms = test_transforms

    def load_cifar10_data(self):
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0,2033, 0.1994, 0.2010))])
        test_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0,2033, 0.1994, 0.2010))])
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=train_transforms)
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
        self.num_cls =  len(train_data.classes)
        self.train_data_classes = train_data.classes
        self.train_data = train_data.data
        self.train_label = train_data.targets
        self.train_data_size = train_data.data.shape[0]
        self.train_transforms = train_transforms
        self.test_data = test_data.data
        self.test_label = test_data.targets
        self.test_data_size = test_data.data.shape[0]
        self.test_transforms = test_transforms

    def mnistDataSetConstruct(self, isIID):
        data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        
        # print("-"*5+"train_images"+"-"*5)
        # 输出第一张图片
        #print(train_images[0].reshape(28,28))
        # print(train_images.shape) # (60000, 28, 28, 1) 一共60000 张图片，每一张是28*28*1
        # print('-'*22+"\n")
        train_labels = extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape) # (60000, 10)
        # print('-'*22+"\n")

        test_images = extract_images(test_images_path)
        # print("-" * 5 + "test_images" + "-" * 5)
        # print(test_images.shape) # (10000, 28, 28, 1)
        # print('-' * 22 + "\n")
        test_labels = extract_labels(test_labels_path)
        # print("-" * 5 + "test_labels" + "-" * 5)
        # print(test_labels.shape) # (10000, 10) 10000维
        #print(train_labels) # 一个对角矩阵
        # print('-' * 22 + "\n")

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        # 训练数据Size
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        # 讲图片每一张图片变成28*28 = 784
        # reshape(60000,28*28)
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        print(train_images.shape) #
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        # 是独立同分布
        '''
            一工有60000个样本
            100个客户端
            IID：
                我们首先将数据集打乱，然后为每个Client分配600个样本。
            Non-IID：
                我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
                然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
        '''
        if isIID:
            #一个参数 默认起点0，步长为1 输出：[0 1 2]
            # a = np.arange(3)
            # 一共60000个
            order = np.arange(self.train_data_size)
            # numpy 中的随机打乱数据方法np.random.shuffle
            '''
                num = np.arange(20)
                print(num)
                # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
                np.random.shuffle(num)
                print(num)
                # [ 1  5 19  9 14  2 12  3  6 18  4  8 16  0 10 17 13  7 15 11]
            '''
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            '''
                # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值。
                two_dim_array = np.array([[1, 3, 5], [0, 4, 3]])
                max_index_axis0 = np.argmax(two_dim_array, axis = 0) # 找 纵向 最大值的下标 
                max_index_axis1 = np.argmax(two_dim_array, axis = 1) # 找 横向 最大值的下标
                print(max_index_axis0)
                print(max_index_axis1)
                
                # [0 1 0] 
                # [2 1]

            '''
            labels = np.argmax(train_labels, axis=1)
            # 对数据标签进行排序

            order = np.argsort(labels)

            print(order.shape)
            print("标签下标排序")
            print(train_labels[order[0:10]])
            self.train_data = train_images[order]
            self.train_label = train_labels[order]


        #print(self.train_label[0:10])

        self.test_data = test_images
        self.test_label = test_labels


    def load_data(self, isIID):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                                 transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
        train_data = train_set.data  # (50000, 32, 32, 3)
        train_labels = train_set.targets
        train_labels = np.array(train_labels)  # 将标签转化为
        print(type(train_labels))  # <class 'numpy.ndarray'>
        print(train_labels.shape)  # (50000,)

        test_data = test_set.data  # 测试数据
        test_labels = test_set.targets
        test_labels = np.array(test_labels)

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        # 将训练集转化为（50000，32*32*3）矩阵
        train_images = train_data.reshape(train_data.shape[0],
                                          train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
        print(train_images.shape)
        # 将测试集转化为（10000，32*32*3）矩阵
        test_images = test_data.reshape(test_data.shape[0],
                                        test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

        # ---------------------------归一化处理------------------------------#
        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        # ----------------------------------------------------------------#
        if isIID:
            # 这里将50000 个训练集随机打乱
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # 按照标签的
            # labels = np.argmax(train_labels, axis=1)
            # 对数据标签进行排序
            order = np.argsort(train_labels)
            print("标签下标排序")
            print(train_labels[order[20000:25000]])
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    # print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__=="__main__":
    print('test data set')
    mnistDataSet = GetDataSet('emnist', 0)
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

