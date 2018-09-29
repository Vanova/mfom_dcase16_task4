import pprint
import src.trainer.dcase_hyper as HP
import src.utils.config as cfg
import src.utils.dirs as dirs
from src.base.pipeline import BasePipeline
import src.data_loader.dcase as DL
from src.data_loader.dcase import DCASEDataLoader
from src.model.cnn_dcase import CNNDcaseModel
from src.model.crnn_dcase import CRNNDcaseModel
from src.trainer.dcase import DCASEModelTrainer, ModelValidator


class DCASEApp(BasePipeline):
    def __init__(self, config, **kwargs):
        super(DCASEApp, self).__init__(config)
        self.pipe_mode = kwargs.get('pipe_mode', 'development')
        self.verbose = kwargs.get('verbose', 2)
        self.overwrite = kwargs.get('overwrite', False)

        self.data_loader = DCASEDataLoader(self.config, pipe_mode=self.pipe_mode)

    def initialize_data(self):
        self.data_loader.initialize()

    def extract_feature(self):
        feat_file = cfg.get_feature_filename(pipe_mode=self.pipe_mode,
                                             path=self.config['path']['features'])
        if not dirs.check_file(feat_file) or self.overwrite:
            self.data_loader.extract_features()
        else:
            print('[INFO] There is feature file: %s' % feat_file)

    def search_hyperparams(self):
        dirs.mkdir(self.config['path']['hyper_search'])
        # initialize default model
        nclass = len(self.data_loader.meta_data.label_names)
        model = self.inflate_model(self.config, nclass, train_mode='pretrain_set')

        hp_srch = HP.HyperSearch(model, self.data_loader, self.config,
                                 fold_id=1,
                                 train_mode='pretrain_set',
                                 verbose=self.verbose,
                                 overwrite=self.overwrite)
        hp_srch.train()

    def system_train(self):
        """
        Pre-training and fine-tuning logic
        """
        print('Model params:')
        pprint.pprint(self.config['model'].toDict())
        dirs.mkdirs(self.config['path']['finetune_set'],
                    self.config['path']['pretrain_set'])

        nclass = len(self.data_loader.meta_data.label_names)

        for fold_id in self.data_loader.meta_data.nfolds:
            if self.config['model']['do_pretrain']:
                print('/*========== Pre-training on FOLD %s ==========*/' % fold_id)

                model = self.inflate_model(self.config, nclass, train_mode='pretrain_set')
                print('Pre-train with loss: %s' % model.model.loss)

                trainer = DCASEModelTrainer(model.model, self.data_loader, self.config,
                                            fold_id=fold_id,
                                            train_mode='pretrain_set',
                                            verbose=self.verbose,
                                            overwrite=self.overwrite)
                trainer.train()

            if self.config['model']['do_finetune']:
                print('/*========== Fine-tuning on FOLD ==========*/' % fold_id)
                mfile = cfg.get_model_filename(path=self.config['path']['pretrain_set'],
                                               train_mode='pretrain_set',
                                               fold=fold_id)
                pre_model = self.inflate_model(self.config, nclass, train_mode='pretrain_set')
                pre_model.load(mfile)

                pre_model.chage_optimizer(self.config['model']['finetune_set'])
                print('Finetune with loss: %s' % pre_model.model.loss)

                finetuner = DCASEModelTrainer(pre_model.model, self.data_loader, self.config,
                                              fold_id=fold_id,
                                              train_mode='finetune_set',
                                              verbose=self.verbose,
                                              overwrite=self.overwrite)
                finetuner.train()

    def system_test(self):
        nclass = len(self.data_loader.meta_data.label_names)

        for fold_id in self.data_loader.meta_data.nfolds:
            for mode in ['pretrain_set', 'finetune_set']:
                print('/*========== Test %s model on FOLD %s ==========*/' % (mode, fold_id))

                model = self.inflate_model(self.config, nclass, train_mode=mode)
                mfile = cfg.get_model_filename(path=self.config['path'][mode],
                                               train_mode=mode,
                                               fold=fold_id)
                model.load(mfile)
                print('Loss: %s' % model.model.loss)

                test_gen = DL.batch_handler(batch_type='validation',
                                            data_file=self.data_loader.feat_file,
                                            fold_lst=self.data_loader.meta_data.fold_list(fold_id, 'test'),
                                            config=self.config['model'][mode],
                                            meta_data=self.data_loader.meta_data)

                history = ModelValidator.validate_model(model=model.model,
                                                        batch_gen=test_gen,
                                                        metrics=self.config['model'][mode]['metrics'])
                print(history)

    @staticmethod
    def inflate_model(config, nclass, train_mode='pretrain_set'):
        model_type = config['model']['type']
        batch_sz = config['model'][train_mode]['batch']
        bands = config['features']['bands']
        frame_wnd = config['model'][train_mode]['context_wnd']
        feat_dim = (batch_sz, bands, frame_wnd, 1)

        if model_type == 'crnn_dcase':
            return CRNNDcaseModel(config['model'][train_mode],
                                  input_shape=feat_dim,
                                  nclass=nclass)
        elif model_type == 'cnn_dcase':
            return CNNDcaseModel(config['model'][train_mode],
                                 input_shape=feat_dim,
                                 nclass=nclass)
        else:
            raise ValueError('[ERROR] Unknown model: %s' % model_type)
