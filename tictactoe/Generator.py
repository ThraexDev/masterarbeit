import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import add

from tictactoe.Board import Board
from tictactoe.Player import Player
from tictactoe.TestPlayer import TestPlayer

input_layer = tf.keras.Input(shape=(27,), name='input')
allowed_moves = tf.keras.Input(shape=(9,), name='allow')
middle = tf.keras.layers.Dense(100, activation=tf.nn.tanh, name='middle')(input_layer)
moveprobability = tf.keras.layers.Dense(9, activation=tf.nn.sigmoid, name='moveprobability')(middle)
winprobability = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='winprobability')(middle)
allowedcardprobability = tf.keras.layers.Multiply(name='finalmoveprobability')([allowed_moves, moveprobability])

model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                            outputs=[allowedcardprobability, winprobability])
model.compile(optimizer='adam',
                   loss=[tf.compat.v2.losses.CategoricalCrossentropy(), tf.compat.v2.losses.mean_squared_error],
                   metrics=['accuracy'])

game_history = []

class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, tfmodel):
        self.model = tfmodel

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        if item % 1000 == 0:
            model.save_weights("model" + str(item))
            won_games = 0
            for test_number in range(0, 100):
                board = Board()
                player0 = Player(model, 1)
                player1 = TestPlayer(0)
                game_not_finished = True
                current_player_won = True
                ai_0_plays = True
                while game_not_finished:
                    ai_0_plays = not ai_0_plays
                    if ai_0_plays:
                        move_probability, player_input, allowed_moves_input = player0.calculate_turn(board)
                        current_player_won, game_not_finished = board.add_move(player0.get_player_number(),
                                                                               int(np.argmax(move_probability)))
                    else:
                        move_probability, player_input, allowed_moves_input = player1.calculate_turn(board)
                        current_player_won, game_not_finished = board.add_move(player1.get_player_number(),
                                                                               int(np.argmax(move_probability)))
                if (ai_0_plays and current_player_won) or (not ai_0_plays and not current_player_won):
                    won_games = won_games + 1
            game_history.append(won_games)
        board = Board()
        player0 = Player(model, 0)
        player1 = Player(model, 1)
        game_not_finished = True
        current_player_won = True
        ai_0_plays = False
        training_input_player0 = []
        training_input_player1 = []
        allowed_moves_player0 = []
        allowed_moves_player1 = []
        move_probabilities_player0 = []
        move_probabilities_player1 = []
        while game_not_finished:
            ai_0_plays = not ai_0_plays
            if ai_0_plays:
                move_probability, player_input, allowed_moves_input = player0.calculate_turn(board)
                training_input_player0.append(player_input)
                allowed_moves_player0.append(allowed_moves_input)
                move_probabilities_player0.append(move_probability)
                current_player_won, game_not_finished = board.add_move(player0.get_player_number(), int(np.argmax(move_probability)))
            else:
                move_probability, player_input, allowed_moves_input = player1.calculate_turn(board)
                training_input_player1.append(player_input)
                allowed_moves_player1.append(allowed_moves_input)
                move_probabilities_player1.append(move_probability)
                current_player_won, game_not_finished = board.add_move(player1.get_player_number(), int(np.argmax(move_probability)))

        if (ai_0_plays and current_player_won) or (not ai_0_plays and not current_player_won):
            win_probabilities_player0 = [[1]] * len(training_input_player0)
            win_probabilities_player1 = [[-1]] * len(training_input_player1)
            #move_probabilities_player1 = [[0.0001] * 9] * len(training_input_player1)
        else:
            win_probabilities_player0 = [[-1]] * len(training_input_player0)
            win_probabilities_player1 = [[1]] * len(training_input_player1)
            #move_probabilities_player0 = [[0.0001] * 9] * len(training_input_player0)
        final_input = []
        final_allow = []
        final_move = []
        final_win = []
        final_input.extend(training_input_player0)
        final_input.extend(training_input_player1)
        final_allow.extend(allowed_moves_player0)
        final_allow.extend(allowed_moves_player1)
        final_move.extend(move_probabilities_player0)
        final_move.extend(move_probabilities_player1)
        final_win.extend(win_probabilities_player0)
        final_win.extend(win_probabilities_player1)
        return {'input': np.array(final_input), 'allow': np.array(final_allow)}, {'finalmoveprobability': np.array(final_move), 'winprobability': np.array(final_win)}


generator = GameStateGenerator(model)

model.fit_generator(generator=generator, epochs=1, workers=1, max_queue_size=1, verbose=1, shuffle=False)

x = list(range(0,len(game_history)))
plt.ylim(top=max(game_history))
plt.ylim(bottom=min(game_history))
plt.ylabel('win rate against base line ai in %')
plt.xlabel('iteration')
plt.plot(x, game_history, color='green', label='win rate')
plt.legend(loc='upper left')
plt.show()
