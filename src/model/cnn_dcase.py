import keras.backend as K
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Convolution2D, Activation, \
    Dropout, Flatten, Input, ELU, BatchNormalization
from keras.optimizers import Adam, SGD
import src.model.mfom as mfom
import src.model.objectives as obj
from src.base.model import BaseModel


class CNNDcaseModel(BaseModel):
    """
    The CNN dim equation: (width - kernel_size + 2*pad)/stride +1
    # Arguments
        input shape: [batch_sz; band; frame_wnd; channel]
    """

    def __init__(self, config, input_shape, nclass):
        super(CNNDcaseModel, self).__init__(config)
        self.input_shape = input_shape
        self.nclass = nclass
        self.build()

    def build(self):
        """
        Construct the main structure of the network
        """
        print('DNN input shape', self.input_shape)

        if K.image_dim_ordering() == 'tf':
            batch_sz, bands, frames, channels = self.input_shape
            assert channels >= 1
            channel_axis = 3
            freq_axis = 1
            nn_shape = (bands, frames, channels)
        else:
            raise NotImplementedError('[ERROR] only for TensorFlow background.')

        kernel_size = (3, 3)
        pool_size = (2, 2)
        nb_filters = self.config['feature_maps']

        # Input block
        feat_input = Input(shape=nn_shape, name='input')
        x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(feat_input)

        # Conv block 1
        x = Convolution2D(nb_filters, 3, 3, border_mode='same', name='conv1')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=pool_size, strides=pool_size, name='pool1')(x)
        x = Dropout(self.config['dropout'], name='dropout1')(x)

        # Conv block 2
        x = Convolution2D(nb_filters, 3, 3, border_mode='same', name='conv2')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=pool_size, strides=pool_size, name='pool2')(x)
        x = Dropout(self.config['dropout'], name='dropout2')(x)

        # Conv block 3
        x = Convolution2D(nb_filters, 3, 3, border_mode='same', name='conv3')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=pool_size, strides=pool_size, name='pool3')(x)
        x = Dropout(self.config['dropout'], name='dropout3')(x)

        # Affine transformation
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation(self.config['activation'])(x)
        x = Dropout(self.config['dropout'])(x)

        # output layer
        x = Dense(self.nclass, name='output_preactivation')(x)
        y_pred = Activation(activation=self.config['out_score'], name='output')(x)

        self._compile_model(input=feat_input, output=y_pred, params=self.config)

    def rebuild(self, new_config):
        pass

    def chage_optimizer(self, new_config):
        """
        Recompile the model with the new loss and optimizer.
        NOTE: Weights are not changed.
        """
        if new_config['freeze_wt']:
            # train only the top layers,
            # i.e. freeze all lower layers
            for layer in self.model.layers[:-4]:
                layer.trainable = False

        # cut MFoM layers: use only output prediction scores
        input = self.model.get_layer(name='input').output
        output = self.model.get_layer(name='output').output
        self._compile_model(input=input, output=output, params=new_config)

    def forward(self, x):
        out_model = self.model
        if self.model.loss in obj.MFOM_OBJECTIVES:
            input = self.model.get_layer(name='input').output
            preact = self.model.get_layer(name='output').output
            out_model = Model(input=input, output=preact)
        return out_model.predict(x)

    def _compile_model(self, input, output, params):
        """
        Compile network structure with particular loss and optimizer
        """
        # ===
        # choose loss
        # ===
        if params['loss'] in obj.MFOM_OBJECTIVES:
            # add 2 layers for Maximal Figure-of-Merit
            y_true = Input(shape=(self.nclass,), name='y_true')
            psi = mfom.UvZMisclassification(name='uvz_misclass')([y_true, output])
            y_pred = mfom.SmoothErrorCounter(name='smooth_error_counter')(psi)

            # MFoM need labels info during training
            input = [y_true, input]
            output = y_pred
            loss = obj.MFOM_OBJECTIVES[params['loss']]
        elif params['loss'] == obj.mfom_eer_embed.__name__:
            loss = obj.mfom_eer_embed
        else:
            loss = params['loss']
        # ===
        # choose optimizer
        # ===
        if params['optimizer'] == 'adam':
            optimizer = Adam(lr=params['learn_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif params['optimizer'] == 'sgd':
            optimizer = SGD(lr=params['learn_rate'], decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optimizer = params['optimizer']

        self.model = Model(input=input, output=output)
        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()
