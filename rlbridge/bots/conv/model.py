from keras import Model
from keras.layers import (Activation, BatchNormalization, Conv1D, Conv2D, Dense,
                          Flatten, Input)
from keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import L2


def flatten_conv(conv_layer, num_channels, reg):
    def chunk(x):
        y = conv_layer(num_channels, 1, kernel_regularizer=L2(reg))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Flatten()(y)
        return y
    return chunk


def flatten_noop():
    def chunk(x):
        return x
    return chunk


def construct_model(
        input_shape,
        structure='2d',
        num_filters=128,
        kernel_size=13,
        num_layers=5,
        flat_channels=4,
        state_size=64,
        hidden_size=64,
        regularization=0.01,
        kernel_reg=0.0,
        aux_outs=None,
):
    game_input = Input(input_shape)

    y = game_input
    conv_layer = Conv2D if structure == '2d' else Conv1D

    for _ in range(num_layers):
        y = conv_layer(
            num_filters,
            kernel_size,
            padding='same',
            kernel_regularizer=L2(kernel_reg)
        )(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

    #for _ in range(num_layers):
    #    y = conv_layer(
    #        num_filters,
    #        (1, kernel_size),
    #        kernel_regularizer=L2(kernel_reg)
    #    )(y)
    #    y = BatchNormalization()(y)
    #    y = Activation('relu')(y)

    if state_size > 0:
        y = Dense(state_size, activation='relu')(Flatten()(y))
        flattener = flatten_noop
    else:
        flattener = partial(flatten_conv, conv_layer, flat_channels, kernel_reg)

    # Call output (39,)
    # same encoding as auction input
    # 0 -> not my turn
    # 1 .. 35 -> 1C through 7NT
    # 36 -> double
    # 37 -> redouble
    # 38 -> pass
    call_flat = flattener()(y)
    call_output = Dense(
        39, name='call_output', activity_regularizer=L2(regularization)
    )(call_flat)

    # Play output (53,)
    # 0 -> not my turn
    # 1..52 -> 2C .. AS
    play_flat = flattener()(y)
    play_output = Dense(
        53, name='play_output', activity_regularizer=L2(regularization)
    )(play_flat)

    value_flat = flattener()(y)
    value_hidden = Dense(
        hidden_size,
        activation='relu',
        kernel_regularizer=L2(regularization)
    )(value_flat)
    value_output = Dense(1, name='value_output')(value_hidden)

    outputs = [call_output, play_output, value_output]
    losses = {
        'call_output': CategoricalCrossentropy(from_logits=True),
        'play_output': CategoricalCrossentropy(from_logits=True),
        'value_output': 'mse',
    }
    loss_weights = {
        'call_output': 1.0,
        'play_output': 1.0,
        'value_output': 0.1,
    }
    if aux_outs is None:
        aux_outs = []
    else:
        aux_outs = aux_outs.split('/')
    if 'contract' in aux_outs:
        contract_flat = flattener()(y)
        contract_output = Dense(5, name='contract_output', activation='relu')(
            contract_flat
        )
        outputs.append(contract_output)
        losses['contract_output'] = 'mse'
        loss_weights['contract_output'] = 1.0
    if 'tricks_won' in aux_outs:
        tricks_flat = flattener()(y)
        tricks_hidden = Dense(hidden_size, activation='relu')(tricks_flat)
        tricks_output = Dense(1, name='tricks_output', activation='relu')(
            tricks_hidden
        )
        outputs.append(tricks_output)
        losses['tricks_output'] = 'mse'
        loss_weights['tricks_output'] = 1.0
    if 'contract_made' in aux_outs:
        made_flat = flattener()(y)
        made_hidden = Dense(hidden_size, activation='relu')(made_flat)
        contract_made_output = Dense(
            1, name='contract_made_output', activation='sigmoid'
        )(made_hidden)
        outputs.append(contract_made_output)
        loss_weights['contract_made_output'] = 1.0

    model = Model(
        inputs=[game_input],
        outputs=outputs,
    )
    model.compile(
        optimizer=SGD(clipnorm=0.5),
        loss=losses,
        loss_weights=loss_weights,
    )
    return model
