from keras import Model
from keras.layers import Conv1D, Dense, Flatten, Input
from keras.optimizers import Adam

from .losses import policy_loss


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
    for i in range(num_layers):
        y = Conv1D(num_filters, kernel_size, activation='relu')(y)

    game_state = Dense(state_size, activation='relu')(Flatten()(y))

    # Call output (39,)
    # same encoding as auction input
    # 0 -> not my turn
    # 1 .. 35 -> 1C through 7NT
    # 36 -> double
    # 37 -> redouble
    # 38 -> pass
    call_hidden = Dense(hidden_size, activation='relu')(game_state)
    call_output = Dense(39, activation='softmax')(call_hidden)

    # Play output (53,)
    # 0 -> not my turn
    # 1..52 -> 2C .. AS
    play_hidden = Dense(hidden_size, activation='relu')(game_state)
    play_output = Dense(53, activation='softmax')(play_hidden)

    value_hidden = Dense(hidden_size, activation='relu')(game_state)
    value_output = Dense(1)(value_hidden)

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
            policy_loss,
            policy_loss,
            'mse'
        ],
        loss_weights=[
            1.0,
            1.0,
            0.02,
        ]
    )
    return model
