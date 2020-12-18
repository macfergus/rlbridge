from keras import Model
from keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Input
from keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import L2


def construct_model(
        input_shape,
        num_filters=128,
        kernel_size=13,
        num_layers=5,
        state_size=64,
        hidden_size=64
):
    game_input = Input(input_shape)

    y = game_input
    for _ in range(num_layers):
        y = Conv1D(num_filters, kernel_size, activation='relu')(y)
        y = BatchNormalization()(y)

    game_state = Dense(state_size, activation='relu')(Flatten()(y))

    # Call output (39,)
    # same encoding as auction input
    # 0 -> not my turn
    # 1 .. 35 -> 1C through 7NT
    # 36 -> double
    # 37 -> redouble
    # 38 -> pass
    call_hidden = Dense(hidden_size, activation='relu')(game_state)
    call_output = Dense(
        39, name='call_output', activity_regularizer=L2(0.01)
    )(call_hidden)

    # Play output (53,)
    # 0 -> not my turn
    # 1..52 -> 2C .. AS
    play_hidden = Dense(hidden_size, activation='relu')(game_state)
    play_output = Dense(
        53, name='play_output', activity_regularizer=L2(0.01)
    )(play_hidden)

    value_hidden = Dense(hidden_size, activation='relu')(game_state)
    value_output = Dense(1, name='value_output')(value_hidden)

    model = Model(
        inputs=[
            game_input,
        ],
        outputs=[
            call_output,
            play_output,
            value_output,
        ]
    )
    model.compile(
        optimizer=Adam(clipnorm=0.5),
        loss=[
            CategoricalCrossentropy(from_logits=True),
            CategoricalCrossentropy(from_logits=True),
            'mse'
        ],
        loss_weights=[
            1.0,
            1.0,
            0.1,
        ]
    )
    return model
