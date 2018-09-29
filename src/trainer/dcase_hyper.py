import pprint
import pickle
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from timeit import default_timer as timer
import src.utils.config as cfg
import src.data_loader.dcase as DL
from src.trainer.dcase import ModelValidator, DCASEModelTrainer

np.random.seed(777)


class HyperSearch(DCASEModelTrainer):
    """
    # Arguments
        model: BaseModel inherited, model for searching hyperparams
        data: BaseDataLoader, generate datasets (for train/test/validate)
        config: DotMap, initial configuration of the 'model',
                by default we choose 'pretrain_set', though these are changing
                during hyper search
        train_mode: String. 'pretrain_set' or 'finetune_set'.
        verbose: Integer. 0, 1, or 2. Verbosity mode.
                    0 = silent, 1 = progress bar, 2 = one line per epoch.
        model: not trained or initially pre-trained model
    """
    MAX_RUNS = 20
    MAX_EPOCHS = 1

    _search_space = {
        'out_score': hp.choice('out_score', ['tanh', 'sigmoid']),
        'dropout': hp.choice('dropout', [0.1, 0.3, 0.5, 0.8]),
        'batch': hp.choice('batch', [8, 16, 32, 64]),
        'learn_rate': hp.choice('learn_rate', [0.1, 0.01, 0.001]),
        'activation': hp.choice('activation', ['relu', 'elu', 'sigmoid', 'tanh']),
        'loss': hp.choice('loss', ['mfom_eer_normalized',
                                   'mfom_microf1',
                                   'mfom_eer_embed',
                                   'mfom_cprim',
                                   'categorical_crossentropy',
                                   'binary_crossentropy',
                                   'mse']),
        'optimizer': hp.choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adadelta'])
    }

    def __init__(self, model, data, config, **kwargs):
        super(HyperSearch, self).__init__(model, data, config, **kwargs)
        self.model_cfg = self.config['model']['pretrain_set']
        self.hp_srch_dir = self.config['path']['hyper_search']
        self.history_file = self.hp_srch_dir + '/history.csv'
        self.trial_file = self.hp_srch_dir + '/trials.pkl'
        self.hp_history = pd.DataFrame()

    def train(self):
        """
        Run trial loop optimizer on search space
        # Return
            save the best value of metrics and hyperparameters
        """
        step = 2  # number of trials every run
        cumulative_trials = 2
        for i in xrange(self.MAX_RUNS // step):
            try:
                trials = pickle.load(open(self.trial_file, 'rb'))
                print('[INFO] Found saved Trials! Loading...')
                cumulative_trials = len(trials.trials) + step
                print('Rerunning from %d trials to %d (+%d) trials' % (len(trials.trials), cumulative_trials, step))
            except:
                trials = Trials()

            fmin(fn=self._experiment,
                 space=self._search_space,
                 algo=tpe.suggest, max_evals=cumulative_trials, trials=trials)

            print('BEST hyperparams so far:')
            sort_res = sorted(trials.results, key=lambda x: x['loss'])
            pprint.pprint(sort_res[:1])

            # save the trials
            with open(self.trial_file, 'wb') as f:
                pickle.dump(trials, f)

            # save trial history
            with open(self.history_file, 'w') as f:
                self.hp_history = pd.DataFrame(sort_res)
                self.hp_history.reset_index(inplace=True, drop=True)
                self.hp_history.to_csv(f)

    def _experiment(self, params):
        """
        Calculate model validation error on the current hyper parameters 'params'.
        Hyper parameters are sampled from the search space.
        NOTE: we have to use only validation data set here
        NOTE2: hyperopt is minimizing, so use 100 - microF1, 100 - Acc, but EER!
        """
        self.model_cfg.update(params)
        print('=*' * 40)
        print('\n Trying Model parameters: ')
        pprint.pprint(self.model_cfg.toDict(), indent=2)

        # rebuild model
        self.model.rebuild(self.model_cfg)

        # init batch generators
        train_gen = DL.batch_handler(batch_type=self.model_cfg['batch_type'],
                                     data_file=self.data.feat_file,
                                     fold_lst=self.data.meta_data.fold_list(self.fold_id, 'train'),
                                     config=self.model_cfg,
                                     meta_data=self.data.meta_data)

        cv_gen = DL.batch_handler(batch_type='validation',
                                  data_file=self.data.feat_file,
                                  fold_lst=self.data.meta_data.fold_list(self.fold_id, 'test'),
                                  config=self.model_cfg,
                                  meta_data=self.data.meta_data)

        samp_sz = train_gen.samples_number()
        print('Epoch size: %d observations' % samp_sz)

        self._init_callbacks(cv_gen)
        batch_sz = self.model_cfg['batch']
        nepo = self.MAX_EPOCHS

        def mfom_batch_wrap(xy_gen):
            for x, y in xy_gen.batch():
                yield [y, x], y

        wrap_gen = mfom_batch_wrap(train_gen) \
            if self.is_mfom_objective(self.model.model) \
            else train_gen.batch()

        # train model
        start = timer()
        self.model.model.fit_generator(wrap_gen,
                                       steps_per_epoch=samp_sz // batch_sz,
                                       nb_epoch=nepo,
                                       verbose=self.verbose,
                                       workers=1,
                                       callbacks=self.callbacks)
        run_time = timer() - start

        # validate model
        mvals = ModelValidator.validate_model(model=self.model.model,
                                              batch_gen=cv_gen,
                                              metrics=self.model_cfg['metrics'])
        train_gen.stop()
        cv_gen.stop()

        loss = mvals[self.config['callback']['monitor']]
        loss = loss if self.config['callback']['mode'] == 'min' else 100. - loss
        r = {'params': self.model_cfg.toDict(),
             'loss': loss,
             'time': run_time,
             'status': STATUS_OK}
        r.update(mvals)
        return r

    def _init_callbacks(self, cv_gen):
        log_dir = cfg.get_log_dir(train_mode='hpsearch')
        print('LOG: %s' % log_dir)

        valtor = ModelValidator(batch_gen=cv_gen,
                                metrics=self.model_cfg['metrics'],
                                monitor=self.config['callback']['monitor'],
                                mode=self.config['callback']['mode'])
        lr_reductor = ReduceLROnPlateau(monitor=self.config['callback']['monitor'],
                                        patience=self.config['callback']['lr_patience'],
                                        verbose=self.verbose,
                                        factor=self.config['callback']['lr_factor'],
                                        min_lr=self.config['callback']['lr_min'])
        tboard = TensorBoard(log_dir=log_dir)
        self.callbacks = [valtor, lr_reductor, tboard]
