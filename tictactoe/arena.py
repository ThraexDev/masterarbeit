import tensorflow as tf
import numpy as np

from tictactoe.Board import Board
from tictactoe.Player import Player
from tictactoe.TestPlayer import TestPlayer

print("Num GPUs Available: ", tf.config.experimental.list_physical_devices())
print(tf.test.is_gpu_available())

input_layer = tf.keras.Input(shape=(27,), name='input')
allowed_moves = tf.keras.Input(shape=(9,), name='allow')
big_layer = tf.keras.layers.Dense(100, activation=tf.nn.tanh, name='big')(input_layer)
middle = tf.keras.layers.Dense(27, activation=tf.nn.tanh, name='middle')(big_layer)
moveprobability = tf.keras.layers.Dense(9, activation=tf.nn.sigmoid, name='moveprobability')(middle)
winprobability = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='winprobability')(middle)
allowedcardprobability = tf.keras.layers.Multiply(name='finalmoveprobability')([allowed_moves, moveprobability])

model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                            outputs=[allowedcardprobability, winprobability])
model.compile(optimizer='Adam',
                   loss=[tf.keras.losses.categorical_crossentropy, tf.compat.v2.losses.mean_squared_error],
                   metrics=['accuracy'])
model.load_weights('resultsfinal/model20000')

class Counter(dict):
    def __missing__(self, key):
       return 0


results = Counter()

won = 0
lost = 0
draw = 0
for test_number in range(0, 1000):
    print(test_number)
    board = Board()
    player0 = Player(model, 0)
    player1 = TestPlayer(1)
    game_not_finished = True
    ai_0_plays = False
    result = ''
    game_feedback = -1
    while game_not_finished:
        ai_0_plays = not ai_0_plays
        if ai_0_plays:
            move_probability, player_input, allowed_moves_input = player0.calculate_turn(board)
            move = int(np.random.choice(9, p=move_probability))
            result = result + str(move) + '-'
            game_feedback, game_not_finished = board.add_move(player0.get_player_number(), move)
        else:
            move_probability, player_input, allowed_moves_input = player1.calculate_turn(board)
            move = int(np.random.choice(9, p=move_probability))
            result = result + str(move) + '-'
            game_feedback, game_not_finished = board.add_move(player1.get_player_number(), move)
    if ai_0_plays:
        if game_feedback == 1:
            result = result + 'won'
            won = won + 1
        else:
            result = result + 'draw'
            draw = draw + 1
    else:
        result = result + 'lost'
        lost = lost + 1
    results[result] += 1

for x in results:
  print(x + ': ' + str(results[x]))
print(won)
print(draw)
print(lost)