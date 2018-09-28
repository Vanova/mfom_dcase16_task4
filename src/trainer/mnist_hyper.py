import pprint
import pickle
import numpy as np
import pandas as pd
from keras import losses
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from timeit import default_timer as timer
import src.model.objectives as obj
from src.model.cnn_mnist import CNNMnistModel
import src.utils.config as cfg
from src.trainer.mnist import ModelValidator

np.random.seed(777)


class CNNMnistHyperSearch(object):
    MAX_RUNS = 20
    MAX_EPOCHS = 10

    _search_space = {
        'out_score': hp.choice('out_score', ['softmax', 'sigmoid']),
        'dropout': hp.choice('dropout', [0.1, 0.3, 0.5, 0.8]),
        'batch': hp.choice('batch', [8, 16, 32, 64]),
        'batch_type': hp.choice('batch_type', ['ivec_stratified', 'ivec_dev']),
        'learn_rate': hp.choice('learn_rate', [0.01]),
        'activation': hp.choice('activation', ['relu',
                                               'elu',
                                               'sigmoid',
                                               'tanh']),
        'loss': hp.choice('loss', ['mfom_eer_normalized',
                                   losses.categorical_crossentropy,
                                   losses.binary_crossentropy,
                                   losses.mse]),
        'optimizer': hp.choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adadelta'])
    }

    def __init__(self, data, config, **kwargs):
        """
        # Arguments
            data: BaseDataLoader
            config: DotMap, configs from yaml file
        """
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = 1

        self.data = data
        self.config = config
        self.overwrite = kwargs['overwrite']
        self.callbacks = []
        self.hp_srch_dir = self.config['path']['hyper_search']
        self.history_file = self.hp_srch_dir + '/history.csv'
        self.trial_file = self.hp_srch_dir + '/trials.pkl'
        self.hp_history = pd.DataFrame()

    def run_search(self):
        """
        Run trial loop optimizer on search space
        # Return
            the best value of metric and hyperparameters
        """
        step = 2  # number of trials every run
        max_trials = 2
        for i in xrange(self.MAX_RUNS // step):
            try:
                trials = pickle.load(open(self.trial_file, 'rb'))
                print('[INFO] Found saved Trials! Loading...')
                max_trials = len(trials.trials) + step
                print('Rerunning from %d trials to %d (+%d) trials' % (len(trials.trials), max_trials, step))
            except:
                trials = Trials()

            fmin(fn=self._experiment,
                 space=self._search_space,
                 algo=tpe.suggest, max_evals=max_trials, trials=trials)

            print('BEST hyperparams so far:')
            sort_res = sorted(trials.results, key=lambda x: x['loss'])
            pprint.pprint(sort_res[:1])

            # save the trials object
            with open(self.trial_file, 'wb') as f:
                pickle.dump(trials, f)

            # save trial history
            with open(self.history_file, 'w') as f:
                # self.hp_history = self.hp_history.append(trials.results[-step:])
                # self.hp_history.sort_values('loss', ascending=True, inplace=True)
                self.hp_history = pd.DataFrame(sort_res)
                self.hp_history.reset_index(inplace=True, drop=True)
                self.hp_history.to_csv(f)

    def _experiment(self, params):
        """
        Calculate model validation error on the current hyper parameters 'params'.
        Hyper parameters are sampled from the search space.
        NOTE: we use only validation set here!
        NOTE2: hyperopt is minimizing, so use 100 - microF1, 1 - Acc, but EER!
        """
        print('=*' * 40)
        print('\n Trying Model parameters: ')
        pprint.pprint(params, indent=2)

        X_train, Y_train = self.data.train_data()
        X_test, Y_test = self.data.test_data()

        model = CNNMnistModel(params)
        if model.model.loss == obj.mfom_eer_normalized:
            X_train = [Y_train, X_train]
            X_test = [Y_test, X_test]

        self._init_callbacks()

        # train model
        start = timer()
        history = model.model.fit(
            X_train, Y_train,
            epochs=self.MAX_EPOCHS,
            verbose=self.verbose,
            batch_size=params['batch'],
            validation_data=(X_test, Y_test),
            callbacks=self.callbacks)
        run_time = timer() - start

        # validate model
        mvals = ModelValidator.validate_model(model=model.model, data=self.data,
                                              metrics=self.config['model']['pretrain_set']['metrics'])

        loss = mvals[self.config['callback']['monitor']]
        loss = loss if self.config['callback']['mode'] == 'min' else 1. - loss
        r = {'params': params,
             'loss': loss,
             'time': run_time,
             'status': STATUS_OK}
        r.update(mvals)
        return r

    def _init_callbacks(self):
        log_dir = cfg.get_log_dir(train_mode='hpsearch')
        print('LOG: %s' % log_dir)

        valtor = ModelValidator(data=self.data,
                                metrics=self.config['model']['pretrain_set']['metrics'],
                                monitor=self.config['callback']['monitor'])
        lr_reductor = ReduceLROnPlateau(monitor=self.config['callback']['monitor'],
                                        patience=self.config['callback']['lr_patience'],
                                        verbose=self.verbose,
                                        factor=self.config['callback']['lr_factor'],
                                        min_lr=self.config['callback']['lr_min'])
        tboard = TensorBoard(log_dir=log_dir)
        self.callbacks = [valtor, lr_reductor, tboard]
