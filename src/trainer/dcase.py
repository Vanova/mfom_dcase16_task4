import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from keras.models import Model
from src.base.trainer import BaseTrainer
import src.utils.config as cfg
import src.utils.dirs as dirs
import src.utils.metrics as mtx
import src.data_loader.dcase as DL
import src.model.objectives as obj


class DCASEModelTrainer(BaseTrainer):
    """
    # Arguments
        model: Keras model, model for training
        data: BaseDataLoader, generate datasets (for train/test/validate)
        config: DotMap, settings for training 'model',
                according to 'training_mode' (i.e. pre-training/finetune)
        train_mode: String. 'pretrain_set' or 'finetune_set'.
        verbose: Integer. 0, 1, or 2. Verbosity mode.
                    0 = silent, 1 = progress bar, 2 = one line per epoch.
        model: not trained or initially pre-trained model
    """

    def __init__(self, model, data, config, **kwargs):
        super(DCASEModelTrainer, self).__init__(model, data, config)

        self.train_mode = kwargs.get('train_mode', 'pretrain_set')
        self.fold_id = kwargs.get('fold_id', 1)
        self.verbose = kwargs.get('verbose', 1)
        self.overwrite = kwargs.get('overwrite', False)

        self.callbacks = []
        self.loss = []
        self.val_loss = []
        self.model_file = cfg.get_model_filename(path=self.config['path'][self.train_mode],
                                                 train_mode=self.train_mode,
                                                 fold=self.fold_id)

    def _init_callbacks(self, cv_gen):
        self.callbacks.append(
            ModelValidator(batch_gen=cv_gen,
                           metrics=self.config['model'][self.train_mode]['metrics'],
                           monitor=self.config['callback']['monitor'],
                           mode=self.config['callback']['mode']))

        self.callbacks.append(
            ModelCheckpoint(
                monitor=self.config['callback']['monitor'],
                filepath=self.model_file,
                mode=self.config['callback']['mode'],
                save_best_only=self.config['callback']['chpt_save_best_only'],
                save_weights_only=self.config['callback']['chpt_save_weights_only'],
                verbose=self.verbose))

        self.callbacks.append(
            ReduceLROnPlateau(monitor=self.config['callback']['monitor'],
                              patience=self.config['callback']['lr_patience'],
                              verbose=self.verbose,
                              factor=self.config['callback']['lr_factor'],
                              min_lr=self.config['callback']['lr_min']))

        self.callbacks.append(
            EarlyStopping(monitor=self.config['callback']['monitor'],
                          patience=self.config['callback']['estop_patience'],
                          verbose=self.verbose,
                          mode=self.config['callback']['mode'],
                          min_delta=0.001))

        log_dir = cfg.get_log_dir(self.train_mode, self.fold_id)
        print('LOG: %s' % log_dir)
        self.callbacks.append(
            TensorBoard(log_dir=log_dir,
                        write_graph=self.config['callback']['tensorboard_write_graph']))

    @staticmethod
    def is_mfom_objective(model):
        return (model.loss in obj.MFOM_OBJECTIVES) or \
               (model.loss in obj.MFOM_OBJECTIVES.values())

    def train(self):
        if not dirs.check_file(self.model_file) or self.overwrite:
            # batch generators
            train_gen = DL.batch_handler(batch_type=self.config['model'][self.train_mode]['batch_type'],
                                         data_file=self.data.feat_file,
                                         fold_lst=self.data.meta_data.fold_list(self.fold_id, 'train'),
                                         config=self.config['model'][self.train_mode],
                                         meta_data=self.data.meta_data)

            cv_gen = DL.batch_handler(batch_type='validation',
                                      data_file=self.data.feat_file,
                                      fold_lst=self.data.meta_data.fold_list(self.fold_id, 'test'),
                                      config=self.config['model'][self.train_mode],
                                      meta_data=self.data.meta_data)

            samp_sz = train_gen.samples_number()
            print('Epoch size: %d observations' % samp_sz)

            self._init_callbacks(cv_gen)
            batch_sz = self.config['model'][self.train_mode]['batch']
            nepo = self.config['model'][self.train_mode]['n_epoch']

            def mfom_batch_wrap(xy_gen):
                for x, y in xy_gen.batch():
                    yield [y, x], y

            wrap_gen = mfom_batch_wrap(train_gen) \
                if self.is_mfom_objective(self.model) \
                else train_gen.batch()

            history = self.model.fit_generator(wrap_gen,
                                               steps_per_epoch=samp_sz // batch_sz,
                                               nb_epoch=nepo,
                                               verbose=self.verbose,
                                               workers=1,
                                               callbacks=self.callbacks)
            self.loss.extend(history.history['loss'])
            self.val_loss.extend(history.history['val_loss'])
            train_gen.stop()
            cv_gen.stop()
        else:
            print('[INFO] There is %s model: %s' % (self.train_mode.upper(), self.model_file))


class ModelValidator(Callback):
    def __init__(self, batch_gen, metrics, monitor, mode):
        super(ModelValidator, self).__init__()
        self.batch_gen = batch_gen
        self.metrics = metrics
        self.monitor = monitor
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best_acc = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_acc = 0.
        else:
            raise AttributeError('[ERROR] ModelValidator mode %s is unknown')

    def on_train_begin(self, logs=None):
        super(ModelValidator, self).on_train_begin(logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' BEFORE TRAINING: Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        super(ModelValidator, self).on_epoch_end(epoch, logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' EPOCH %d. Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            epoch, vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        super(ModelValidator, self).on_train_end(logs)
        print('=' * 20 + ' Training report ' + '=' * 20)
        print('Best validation %s: epoch %s / %.4f\n' % (self.monitor.upper(), self.best_epoch, self.best_acc))

    @staticmethod
    def validate_model(model, batch_gen, metrics):
        """
        # Arguments
            model: Keras model
            data: BaseDataLoader
            metrics: list of metrics
        # Output
            dictionary with values of metrics and loss
        """
        cut_model = model
        if DCASEModelTrainer.is_mfom_objective(model):
            input = model.get_layer(name='input').output
            preact = model.get_layer(name='output').output
            cut_model = Model(input=input, output=preact)

        n_class = cut_model.output_shape[1]
        y_true, y_pred = np.empty((0, n_class)), np.empty((0, n_class))
        loss, cnt = 0, 0

        for X_b, Y_b in batch_gen.batch():
            ps = cut_model.predict_on_batch(X_b)
            y_pred = np.vstack([y_pred, ps])
            y_true = np.vstack([y_true, Y_b])
            # NOTE: it is fake loss, caz Y is fed
            if DCASEModelTrainer.is_mfom_objective(model):
                X_b = [Y_b, X_b]
            l = model.test_on_batch(X_b, Y_b)
            loss += l
            cnt += 1

        vals = {'val_loss': loss / cnt}

        for m in metrics:
            if m == 'micro_f1':
                p = mtx.step(y_pred, threshold=0.5)
                vals[m] = mtx.micro_f1(y_true, p)
            elif m == 'pooled_eer':
                p = y_pred.flatten()
                y = y_true.flatten()
                vals[m] = mtx.eer(y, p)
            elif m == 'class_wise_eer':
                vals[m] = np.mean(mtx.class_wise_eer(y_true, y_pred))
            elif m == 'accuracy':
                p = np.argmax(y_pred, axis=-1)
                y = np.argmax(y_true, axis=-1)
                vals[m] = mtx.pooled_accuracy(y, p)
            else:
                raise KeyError('[ERROR] Such a metric is not implemented: %s...' % m)
        return vals
