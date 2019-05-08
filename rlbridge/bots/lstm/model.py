import numpy as np
from keras import Model
from keras.layers import (LSTM, Concatenate, Dense, Flatten, Input, Reshape,
                          TimeDistributed)
from keras.optimizers import Adam

from ...rl import policy_loss

# The longest possible auction has 319 calls. But that is very
# unrealistic. Here we impose an arbitrary cap of 60 calls.
MAX_GAME = 1 + 60 + 52


def construct_model(input_shape, lstm_size=512, lstm_depth=2):
    game_input = Input(batch_shape=(1,1) + input_shape)

    lstm_y = game_input
    for i in range(lstm_depth):
        # The inner layers should return sequences; the last layer does
        # not need to.
        seq = i != lstm_depth - 1
        lstm_y = LSTM(lstm_size, return_sequences=seq, stateful=True)(lstm_y)

    # Call output (39,)
    # same encoding as auction input
    # 0 -> not my turn
    # 1 .. 35 -> 1C through 7NT
    # 36 -> double
    # 37 -> redouble
    # 38 -> pass
    call_output = Dense(39, activation='softmax')(lstm_y)

    # Play output (53,)
    # 0 -> not my turn
    # 1..52 -> 2C .. AS
    play_output = Dense(53, activation='softmax')(lstm_y)

    model = Model(
        inputs=[
            game_input,
        ],
        outputs=[
            call_output,
            play_output,
        ]
    )
    model.compile(
        optimizer=Adam(clipnorm=0.5),
        loss=policy_loss
    )
    return model
