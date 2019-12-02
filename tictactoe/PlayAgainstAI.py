import numpy as np
import tensorflow as tf

from tictactoe.Board import Board
from tictactoe.Player import Player

input_layer = tf.keras.Input(shape=(18,), name='input')
allowed_moves = tf.keras.Input(shape=(9,), name='allow')
middle = tf.keras.layers.Dense(100, activation=tf.nn.tanh, name='middle')(input_layer)
moveprobability = tf.keras.layers.Dense(9, activation=tf.nn.sigmoid, name='moveprobability')(middle)
winprobability = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='winprobability')(middle)
allowedcardprobability = tf.keras.layers.Multiply(name='finalmoveprobability')([allowed_moves, moveprobability])

model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                       outputs=[allowedcardprobability, winprobability])
model.compile(optimizer='adam',
              loss=[tf.compat.v2.losses.BinaryCrossentropy(), tf.compat.v2.losses.mean_squared_error],
              metrics=['accuracy'])
model.load_weights("starter_4")

ai_player = Player(model, 0)
board = Board()
game_not_finished = True
ai_turn = True
current_player_won = False

while game_not_finished:
    if ai_turn:
        prob, input_ai, allow = ai_player.calculate_turn(board)
        print(prob)
        print(input_ai)
        print(allow)
        move = np.argmax(prob)
        current_player_won, game_not_finished = board.add_move(0, move)
    else:
        print(board.get_allow())
        move = int(input('move: '))
        current_player_won, game_not_finished = board.add_move(1, move)
    ai_turn = not ai_turn

if current_player_won:
    print('current player won')
