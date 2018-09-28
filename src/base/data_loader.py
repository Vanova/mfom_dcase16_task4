"""
Classes for working with datasets:
 load data, meta data, data batches generator
"""
import h5py
import numpy as np


class BaseDataLoader(object):
    """
    Abstract class for accessing data: meta info,
    processing and generating data
    """

    def __init__(self, config):
        self.config = config

    @property
    def meta_data(self):
        raise NotImplementedError

    def initialize(self):
        """
        Load data, prepare data meta information
        """
        raise NotImplementedError

    def extract_features(self):
        raise NotImplementedError

    def train_data(self):
        """
        Lazy method, only for small datasets
        # Return
            x, y - whole data
        """
        raise NotImplementedError

    def test_data(self):
        """
        Lazy method, only for small datasets
        # Return
            x, y - whole data
        """
        raise NotImplementedError

    def eval_data(self):
        """
        Lazy method, only for small datasets
        # Return
            x - whole data without labels
        """
        raise NotImplementedError


class BaseMeta(object):
    DATA_TYPES = ['train', 'validation', 'test', 'evaluation']
    LABEL_SEPARATOR = ';'

    @property
    def label_names(self):
        raise NotImplementedError

    @property
    def nfolds(self):
        raise NotImplementedError

    def fold_list(self, fold, data_type):
        raise NotImplementedError

    def file_list(self):
        raise NotImplementedError

    def labels_str_encode(self, lab_dig):
        raise NotImplementedError

    def labels_dig_encode(self, lab_str):
        raise NotImplementedError


class BaseGenerator(object):
    def __init__(self, data_file, batch_sz=1, window=1, fold_list=None):
        self.data_file = data_file
        self.batch_sz = batch_sz
        self.window = window
        self.fold_list = fold_list

        self.hdf = h5py.File(self.data_file, 'r')
        if self.fold_list:
            print('Generate from the FOLD list: %d files' % len(set(self.fold_list[0])))
        else:
            self.fold_list = list(self.hdf.keys())
            print('Generate from the WHOLE dataset: %d files' % len(self.fold_list))

    def batch(self):
        raise NotImplementedError

    def batch_shape(self):
        raise NotImplementedError

    def samples_number(self):
        raise NotImplementedError

    def stop(self):
        self.hdf.close()

# class BaseGenerator(object):
#     """
#     Generate mini-batches from features stored in HDF5 file.
#
#     hdf5 storage of datasets with attributes as labels:
#     hdf5 datasets (2D[3D] arrays) [bands; frames; [channel]]
#     with attributes as labels (1D array)
#
#     # Arguments
#         data_file: String. hdf5 feature full file path
#         meta_data: BaseMeta.
#         batch_sz: Integer.
#     """
#     def __init__(self, data_file, batch_sz=1, file_list=None, meta_data=None, window=[0, 0]):
#         self.hdf = h5py.File(data_file, 'r')
#         self.batch_sz = batch_sz
#         self.file_list = file_list
#         self.meta_data = meta_data
#         # context frame window
#         self.window = window
#
#         if self.file_list:
#             print('Generate from list: %d files' % len(self.file_list))
#         else:
#             self.file_list = list(self.hdf.keys())
#             print('Files in dataset: %d files' % len(self.file_list))
#
#         if not self.meta_data:
#             self.data_stat = self._calculate_stat()
#
#     def batch(self):
#         raise NotImplementedError
#
#     def batch_shape(self):
#         """
#         NOTE: input dataset always is in Tensorflow order [bands, frames, channels]
#         # Return:
#             Tensorflow:
#                 3D data [batch_sz; band; frame_wnd; channel]
#                 2D data [batch_sz; band; frame_wnd]
#                 1D data [dim]
#         """
#         sh = np.array(self.hdf[self.file_list[0]]).shape
#         if len(sh) == 3:
#             bands, _, channels = sh
#             assert channels >= 1
#             return self.batch_sz, bands, self.window, channels
#         elif len(sh) == 2:
#             bands, _ = sh
#             return self.batch_sz, bands, self.window
#         elif len(sh) == 1:
#             dim = sh[0]
#             return self.batch_sz, dim
#
#     def samples_size(self):
#         """
#         # Return:
#             number of total observations, i.e. N_files * file_len/wnd
#         """
#         total_smp = 0
#         for ifn, fn in enumerate(self.file_list):
#             dim, N, ch = self.hdf[fn].shape
#             total_smp += N // self.window
#         return total_smp // self.batch_sz * self.batch_sz
#
#     def stop(self):
#         self.hdf.close()
#
#     def _calculate_stat(self):
#         raise NotImplementedError
