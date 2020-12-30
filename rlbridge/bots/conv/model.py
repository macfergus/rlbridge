from keras import Model
from keras.layers import (Activation, BatchNormalization, Conv1D, Dense,
                          Flatten, Input)
from keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import L2


def construct_model(
        input_shape,
        num_filters=128,
        kernel_size=13,
        num_layers=5,
        state_size=64,
        hidden_size=64,
        regularization=0.01,
        kernel_reg=0.0,
        aux_outs=None,
):
    game_input = Input(input_shape)

    y = game_input
    for _ in range(num_layers):
        y = Conv1D(
            num_filters, kernel_size, padding='same',
            kernel_regularizer=L2(kernel_reg) if kernel_reg > 0 else None
        )(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

    game_state = Dense(state_size, activation='relu')(Flatten()(y))

    # Call output (39,)
    # same encoding as auction input
    # 0 -> not my turn
    # 1 .. 35 -> 1C through 7NT
    # 36 -> double
    # 37 -> redouble
    # 38 -> pass
    call_hidden = Dense(
        hidden_size,
        activation='relu',
        kernel_regularizer=L2(regularization)
    )(game_state)
    call_output = Dense(
        39, name='call_output', activity_regularizer=L2(regularization)
    )(call_hidden)

    # Play output (53,)
    # 0 -> not my turn
    # 1..52 -> 2C .. AS
    play_hidden = Dense(
        hidden_size,
        activation='relu',
        kernel_regularizer=L2(regularization)
    )(game_state)
    play_output = Dense(
        53, name='play_output', activity_regularizer=L2(0.01)
    )(play_hidden)

    value_hidden = Dense(
        hidden_size,
        activation='relu',
        kernel_regularizer=L2(regularization)
    )(game_state)
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
        contract_output = Dense(5, name='contract_output', activation='relu')(
            game_state
        )
        outputs.append(contract_output)
        losses['contract_output'] = 'mse'
        loss_weights['contract_output'] = 1.0

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
