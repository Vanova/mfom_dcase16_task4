import keras.backend as K
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Convolution2D, Activation, \
    Dropout, Reshape, Input, ELU, BatchNormalization, GRU
from keras.optimizers import Adam, SGD
import src.model.mfom as mfom
import src.model.objectives as obj
from src.base.model import BaseModel


class CRNNDcaseModel(BaseModel):
    """
    The CNN dim equation: (width - kernel_size + 2*pad)/stride +1
    # Arguments
        input shape: [batch_sz; band; frame_wnd; channel]
    """

    def __init__(self, config, input_shape, nclass):
        super(CRNNDcaseModel, self).__init__(config)
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
            raise NotImplementedError('[ERROR] Only for TensorFlow background.')

        nb_filters = self.config['feature_maps']

        # Input block
        feat_input = Input(shape=nn_shape, name='input')
        x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(feat_input)

        # Conv block 1
        x = Convolution2D(nb_filters, 3, 3, border_mode='same', name='conv1')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
        x = Dropout(self.config['dropout'], name='dropout1')(x)

        # Conv block 2
        x = Convolution2D(nb_filters, 3, 3, border_mode='same', name='conv2')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool2')(x)
        x = Dropout(self.config['dropout'], name='dropout2')(x)

        # Conv block 3
        x = Convolution2D(2 * nb_filters, 3, 3, border_mode='same', name='conv3')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), name='pool3')(x)
        x = Dropout(self.config['dropout'], name='dropout3')(x)

        # Conv block 4
        x = Convolution2D(2 * nb_filters, 3, 3, border_mode='same', name='conv4')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool4')(x)
        x = Dropout(self.config['dropout'], name='dropout4')(x)

        x = Reshape((-1, 2 * nb_filters))(x)

        # GRU block 1, 2, output
        # x = GRU(32, return_sequences=True, name='gru1')(x)
        x = GRU(32, return_sequences=False, name='gru2')(x)
        x = Dropout(self.config['dropout'])(x)

        # Affine transformation
        x = Dense(self.nclass, name='output_preactivation')(x)
        y_pred = Activation(activation=self.config['out_score'], name='output')(x)

        self._compile_model(input=feat_input, output=y_pred, params=self.config)

    def rebuild(self, new_config):
        """
        Recompile the model with the new hyper parameters.
        NOTE: network topology is changing according to the 'new_config'
        """
        self.config.update(new_config)
        self.build()

    def chage_optimizer(self, new_config):
        """
        Recompile the model with the new loss and optimizer.
        NOTE: network topology is not changing.
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
