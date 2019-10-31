from .data_manager import DataManager
import os
from tqdm import tqdm
import gzip
import struct
import urllib
import numpy as np


class MNISTDataManager(DataManager):
    """ Data manger for MNIST data

    Attributes:
        config(Config): configurations of params
        data_set(tuple): (labels, images) for later iterating
    """
    def __init__(self, config):
        super(MNISTDataManager, self).__init__()
        self.config = config
        self.data_set = self._read_data(label_url=self.config.label_url,
                                        image_url=self.config.image_url)

    def _read_data(self, **source):
        """ Read MNIST data set will there special format

        Args:
            label_url(str): url for label data
            image_url(str): url for image data

        Returns:
            label(array): (total_num, labels)
            image(array): (total_num, H, W)
        """
        label_url = source['label_url']
        image_url = source['image_url']
        with gzip.open(self._download(label_url, self.config.local_data_dir)) as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        with gzip.open(self._download(image_url, self.config.local_data_dir), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
        return (label, image)

    def _download(self, url, local_data_dir, force_download=False):
        """ Download if there's no data in local path

        Args:
            url: url for the data
            local_data_dir: data where data residents
            force_download: download no matter local existence of data

        Returns:
            filepath(str): path to the file
        """
        filepath = local_data_dir + url.split("/")[-1]
        if force_download or not os.path.exists(filepath):
            with tqdm(unit='B', unit_scale=True, leave=True, desc=filepath)as pbar:
                urllib.request.urlretrieve(url, filepath, self._tqdm_hook(pbar))
        return filepath

    def _tqdm_hook(self, t):
        """ TODO: figure out what's under hood"""
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            """
            Args:
                b: downloaded blocks count
                bsize: block size
                tsize: total size
            """
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    def __iter__(self, *args, **kwargs):
        """ Return tuple of one label and one image"""
        for i in range(len(self.data_set[0])):
            yield (self.data_set[0][i], self.data_set[1][i])

    def mini_batch(self, batch_size):
        """ See detailed documentation in base class"""
        def reset():
            """ Clear the code """
            _labels = []
            _images = []
            _sample_num = 0
            return _labels, _images, _sample_num

        labels, images, sample_num = reset()
        for sample in self:
            if sample_num == batch_size:
                yield (labels, images)
                labels, images, sample_num = reset()
            labels.append(sample[0])
            images.append(sample[1])
            sample_num += 1
