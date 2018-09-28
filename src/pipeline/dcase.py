import pprint
import src.trainer.dcase_hyper as HP
import src.utils.config as cfg
import src.utils.dirs as dirs
from src.base.pipeline import BasePipeline
from src.data_loader.dcase import DCASEDataLoader
from src.model.cnn_dcase import CNNDcaseModel
from src.trainer.dcase import DCASEModelTrainer


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
        # initialize default model
        # def_model = MLPDcaseModel(self.config['model']['pretrain_set'])
        dirs.mkdir(self.config['path']['hyper_search'])
        hp_srch = HP.MLPDcaseHyperSearch(data=self.data_loader,
                                         config=self.config,
                                         verbose=self.verbose,
                                         overwrite=self.overwrite)
        res = hp_srch.run_search()
        print(res)

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
                print('/*========== Pre-training ==========*/')
                batch_sz = self.config['model']['pretrain_set']['batch']
                bands = self.config['features']['bands']
                frame_wnd = self.config['model']['pretrain_set']['context_wnd']
                feat_dim = (batch_sz, bands, frame_wnd, 1)

                model = CNNDcaseModel(self.config['model']['pretrain_set'],
                                      input_shape=feat_dim,
                                      nclass=nclass)
                print('Pre-train with loss: %s' % model.model.loss)

                trainer = DCASEModelTrainer(model.model, self.data_loader, self.config,
                                            fold_id=fold_id,
                                            train_mode='pretrain_set',
                                            verbose=self.verbose,
                                            overwrite=self.overwrite)
                trainer.train()

            if self.config['model']['do_finetune']:
                print('/*========== Fine-tuning ==========*/')
                mfile = cfg.get_model_filename(path=self.config['path']['pretrain_set'],
                                               train_mode='pretrain_set',
                                               fold=fold_id)

                batch_sz = self.config['model']['finetune_set']['batch']
                bands = self.config['features']['bands']
                frame_wnd = self.config['model']['pretrain_set']['context_wnd']
                feat_dim = (batch_sz, bands, frame_wnd, 1)

                pre_model = CNNDcaseModel(self.config['model']['pretrain_set'],
                                          input_shape=feat_dim,
                                          nclass=nclass)
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
        print('[TODO] test system: store the testing prediction scores...')

    def system_evaluate(self):
        print('[TODO] evaluate system: store the prediction scores...')
