import pandas as pd
import numpy as np
import os.path as path
from p_tqdm import p_imap
from sklearn.preprocessing import MultiLabelBinarizer
import src.utils.io as io
import src.utils.dirs as dirs
import src.utils.config as cfg
from src.base.data_loader import BaseDataLoader, BaseMeta, BaseGenerator
import src.features.speech as F


class DCASEDataLoader(BaseDataLoader):
    """
    Dataset handler facade.
    # Arguments
        config: dict. Parameters of data loader.
        pipe_mode: String. 'development' or 'submission'
    """

    def __init__(self, config, pipe_mode, **kwargs):
        super(DCASEDataLoader, self).__init__(config)
        self.pipe_mode = pipe_mode
        self.overwrite = kwargs.get('overwrite', False)
        self.feat_file = cfg.get_feature_filename(pipe_mode=pipe_mode,
                                                  path=self.config['path']['features'])

        dirs.mkdirs(self.config['path']['meta'],
                    self.config['path']['features'])
        self._meta = None

    @property
    def meta_data(self):
        return self._meta

    def initialize(self):
        print('[INFO] Initialize meta data...')
        if dirs.isempty(self.config['path']['meta']) or self.overwrite:
            self._prepare_meta_lists()

        if self.pipe_mode == cfg.PipeMode.DEV:
            self._meta = DCASE2016Task4Dev(data_dir=self.config['experiment']['development_dataset'],
                                           feat_conf=self.config['features'],
                                           meta_dir=self.config['path']['meta'])
        elif self.pipe_mode == cfg.PipeMode.SUBMIT:
            self._meta = DCASE2016Task4Eval(data_dir=self.config['experiment']['development_dataset'],
                                            feat_conf=self.config['features'],
                                            meta_dir=self.config['path']['meta'])

    def extract_features(self):
        print('[INFO] Extract features from all audio files')

        all_lists = self.meta_data.file_list()
        fnames = list(set(all_lists[0]))

        # prepare extractor
        feat_type = self.config['features']['type']
        extractor = F.prepare_extractor(feats=feat_type, params=self.config['features'])
        writer = io.HDFWriter(file_name=self.feat_file)

        iterator = p_imap(lambda fn: self._extraction_job(fn, extractor, self.meta_data.data_dir), fnames)

        for result in iterator:
            for fn, feat in result.items():
                fid = path.basename(fn)
                writer.append(file_id=fid, feat=feat)
        writer.close()
        del writer
        print('Files processed: %d' % len(fnames))

    def train_data(self):
        raise NotImplementedError('[INFO] This is lazy method for small datasets.'
                                  'Use MetaData and BaseGenerator.')

    def test_data(self):
        raise NotImplementedError('[INFO] This is lazy method for small datasets.'
                                  'Use MetaData and BaseGenerator.')

    def eval_data(self):
        raise NotImplementedError('[INFO] This is lazy method for small datasets.'
                                  'Use MetaData and BaseGenerator.')

    def _prepare_meta_lists(self):
        lists_dir = self.config['experiment']['lists_dir']
        folds = range(1, 6)
        data_sets = dict(test='evaluate', train='train')
        for dset, fname in data_sets.items():
            for i in folds:
                fn = path.join(lists_dir, 'fold%d_%s.txt' % (i, fname))
                pd_data = io.load_csv(file_name=fn, col_name=['file', 'env', 'labs_name', 'labs'], delim='\t')
                pd_data['labs'] = pd_data['labs'].fillna('S')
                # dump
                dmp_file = path.join(self.config['path']['meta'], 'fold%d_%s.csv' % (i, dset))
                io.save_csv(dmp_file, list_data=zip(pd_data['file'].values, pd_data['labs'].values), delim='\t')

    def _extraction_job(self, fname, extractor, data_dir):
        file_path = path.join(data_dir, fname)
        x, fs = F.load_sound_file(file_path)
        feat = extractor.extract(x, fs)
        return {fname: feat}


def batch_handler(batch_type, data_file, config, fold_lst=None, meta_data=None):
    """
    batch_type:
        'seq_slide_wnd' - cut sequential chunks from the taken file
        'rnd_wnd' - cut random chunk from the file, then choose another file
        'eval' - slice one file and return samples to do evaluation
    config: current model configuration
    """
    batch_sz = config['batch']
    wnd = config['context_wnd']
    print('Batch type: %s' % batch_type)

    if batch_type == 'seq_slide_wnd':
        return SequentialGenerator(data_file=data_file,
                                   batch_sz=batch_sz,
                                   window=wnd,
                                   fold_list=fold_lst,
                                   meta_data=meta_data,
                                   config=config,
                                   shuffle=False)
    elif batch_type == 'validation':
        return ValidationGenerator(data_file=data_file,
                                   batch_sz=batch_sz,
                                   window=wnd,
                                   fold_list=fold_lst,
                                   meta_data=meta_data,
                                   config=config,
                                   shuffle=False)
    else:
        raise ValueError('Unknown batch type [' + batch_type + ']')


class SequentialGenerator(BaseGenerator):
    """
    Generate sequence of observations.
    # Output
        Slice HDF5 datasets and return batch: [smp x bands x frames x channel]
    """

    def __init__(self, data_file, batch_sz=1, window=0, fold_list=None, meta_data=None, **kwargs):
        super(SequentialGenerator, self).__init__(data_file, batch_sz, window, fold_list)
        self.meta_data = meta_data  # meta is needed if we need to calculate data stats
        self.config = kwargs['config']  # model config: know objective function
        self.hop_frames = self.window

    def batch(self):
        while 1:
            # iterate over datasets until the batch is filled up
            count_smp = 0
            X, Y = [], []
            fnames = list(set(self.fold_list[0]))
            np.random.shuffle(fnames)
            for fn, lab in zip(*self.fold_list):
                fid = path.basename(fn)
                feat = np.array(self.hdf[fid], dtype=np.float32)
                dim, n_frames, channel = feat.shape
                last = n_frames // self.hop_frames * self.hop_frames

                for start in range(0, last, self.hop_frames):
                    if count_smp < self.batch_sz:
                        count_smp += 1
                    else:
                        yield np.array(X), np.array(Y)
                        count_smp = 1
                        X, Y = [], []
                    X.append(feat[:, start:start + self.window, :])
                    Y.append(lab)

    def batch_shape(self):
        """
        NOTE: dataset is always in Tensorflow order [bands, frames, channels]
        # Return:
            ndarray [batch_sz; band; frame_wnd; channel]
        """
        fn = path.basename(self.fold_list[0][0])
        sh = np.array(self.hdf[fn]).shape
        bands, _, channels = sh
        assert channels >= 1
        return self.batch_sz, bands, self.window, channels

    def samples_number(self):
        """
        # Output
            number of total observations, i.e. N_files * file_length/hop_wnd
        """
        total_smp = 0
        for fn in set(self.fold_list[0]):
            dim, n_frames, ch = self.hdf[path.basename(fn)].shape
            total_smp += n_frames // self.hop_frames
        return total_smp


class ValidationGenerator(SequentialGenerator):
    def batch(self):
        count_smp = 0
        X, Y = [], []
        for fn, lab in zip(*self.fold_list):
            fid = path.basename(fn)
            feat = np.array(self.hdf[fid], dtype=np.float32)
            dim, n_frames, channel = feat.shape
            last = n_frames // self.hop_frames * self.hop_frames

            if last == 0:
                print("[INFO] file %s shorter than context frame window: %d" % (fn, n_frames))
            else:
                for start in xrange(0, last, self.hop_frames):
                    if count_smp < self.batch_sz:
                        count_smp += 1
                    else:
                        yield np.array(X), np.array(Y)
                        count_smp = 1
                        X, Y = [], []
                    X.append(feat[:, start:start + self.window, :])
                    Y.append(lab)


class DCASE2016Task4Dev(BaseMeta):
    """
    CHiME-Home Development dataset meta lists.
    We use the same 5 folds lists as in the DCASE 2016: Task 4 challenge.

    Meta data format: we use csv format
        ['file' 'class_label']
        e.g. [path/file.wav  lab1;lab2;lab3], can be multi-label format

    # Arguments
        data_dir: path to the dataset
        meta_dir: path of the processed metadata lists
        feat_conf: DotMap. Feature configuration
    """
    DATA_TYPES = ['train', 'test']  # 'validation', 'test']

    # DATA_TYPES = dict(train='fold%d_train.txt',
    #                   val='fold%d_validation.txt',
    #                   test='fold%d_test.txt')

    def __init__(self, data_dir, feat_conf, meta_dir):
        self.data_dir = data_dir
        self.feat_conf = feat_conf
        self.meta_dir = meta_dir

        self.col = ['file', 'class_label']
        self._meta_file_template = 'fold{num}_{dtype}.csv'
        self.folds_num = 5
        self.lencoder = self._labels_encoder()

    @property
    def label_names(self):
        return list(self.lencoder.classes_)

    @property
    def nfolds(self):
        """Return list of folds indices"""
        return range(1, self.folds_num + 1)

    def labels_str_encode(self, lab_dig):
        """
        Transform hot-vector to 'class_label' format
        lab_dig: list.
        """
        return list(self.lencoder.inverse_transform(lab_dig))

    def labels_dig_encode(self, lab_str):
        """
        Transform 'class_label' to hot-vector format
        lab_str: list.
        """
        return self.lencoder.transform(lab_str)

    def fold_list(self, fold, data_type):
        """
        fold: Integer. Number of fold to return
        data_type: String. 'train', 'validation' or 'test' for development lists
        return: list. File names and labels hot vector
        """
        if not (data_type in self.DATA_TYPES):
            raise AttributeError('[ERROR] No dataset type: %s' % data_type)
        flist = self._load_fold_list(fold, data_type)
        return self._format_list(flist)  # file_name, hot_vecs

    def file_list(self):
        """
        Merge file lists of the development dataset (training + validation + test)
        return: list. File names and labels hot vector
        """
        mrg = pd.DataFrame()
        for dt in self.DATA_TYPES:
            fl = self._load_fold_list(fold=1, data_type=dt)
            mrg = mrg.append(fl)
        mrg.reset_index(drop=True, inplace=True)
        return self._format_list(mrg)  # file_name, hot_vecs

    def _labels_encoder(self):
        """
        prepare labels encoder from string to digits
        """
        # fn = self._meta_file_template.format(num=1, dtype=self.DATA_TYPES[0])
        # meta_file = path.join(self.meta_dir, fn)
        # pd_meta = io.load_csv(meta_file, col_name=self.col, delim='\t')
        pd_meta = self._load_fold_list(fold=1, data_type=self.DATA_TYPES[0])
        le = MultiLabelBinarizer()
        labels_list = pd_meta['class_label'].astype(str)
        labels_list = labels_list.str.split(self.LABEL_SEPARATOR)
        le.fit(labels_list)
        return le

    def _load_fold_list(self, fold, data_type):
        fn = self._meta_file_template.format(num=fold, dtype=data_type)
        meta_file = path.join(self.meta_dir, fn)
        pd_meta = io.load_csv(meta_file, col_name=self.col, delim='\t')
        return pd_meta

    def _format_list(self, df):
        fnames = list(df['file'])
        # labels to hot-vectors
        labs = df['class_label'].str.split(self.LABEL_SEPARATOR)
        hot_vecs = self.labels_dig_encode(labs)
        return fnames, hot_vecs


class DCASE2016Task4Eval(BaseMeta):
    """
    CHiME-Home Evaluation dataset meta lists.
    We use the same 5 folds lists as in the DCASE 2016: Task 4 challenge.

    Meta data format: list of files without labels

    # Arguments
        data_dir: path to the dataset
        meta_dir: path of the processed metadata lists
        feat_conf: DotMap. Feature configuration
    """
    DATA_TYPES = ['evaluation']

    def __init__(self, data_dir, feat_conf, meta_dir):
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.feat_conf = feat_conf

        self.col = ['file']
        self.folds_num = 1

    def file_list(self):
        """
        return: file list of the evaluation dataset
        """
        # parse list of files from evaluation_setup
        meta_file = path.join(self.meta_dir, 'evaluation.csv')
        pd_meta = io.load_csv(meta_file, col_name=self.col)
        return pd_meta['file'].values.tolist()

    def nfolds(self):
        return range(1, self.folds_num + 1)
