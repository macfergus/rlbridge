import numpy as np
from keras import Model
from keras.layers import (LSTM, Concatenate, Dense, Flatten, Input, Reshape,
                          TimeDistributed)


def construct_model(input_shape, lstm_size=512, lstm_depth=2):
    game_input = Input(shape=input_shape)

    shaped_input = Reshape((1, -1))(game_input)
    lstm_y = shaped_input
    for _ in range(lstm_depth):
        lstm_y = LSTM(lstm_size, return_sequences=True)(lstm_y)

    # Call output (39,)
    # same encoding as auction input
    # 0 -> not my turn
    # 1 .. 35 -> 1C through 7NT
    # 36 -> double
    # 37 -> redouble
    # 38 -> pass
    call_output = TimeDistributed(Dense(39, activation='softmax'))(lstm_y)

    # Play output (53,)
    # 0 -> not my turn
    # 1..52 -> 2C .. AS
    play_output = TimeDistributed(Dense(53, activation='softmax'))(lstm_y)

    model = Model(
        inputs=[
            game_input,
        ],
        outputs=[
            call_output,
            play_output,
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model
