import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import add

from nimmt.Board import Board
from nimmt.Player import Player
from nimmt.TestPlayer import TestPlayer

print("Num GPUs Available: ", tf.config.experimental.list_physical_devices())
print(tf.test.is_gpu_available())

cardshandedtoeachplayer = 10
playeramount = 3
cardamount = 61
amountofbatches = 3
maxbatchcards = 5

input_layer = tf.keras.Input(shape=(435,), name='input')
allowed_moves = tf.keras.Input(shape=(cardamount*amountofbatches,), name='allow')
big_layer = tf.keras.layers.Dense(800, activation=tf.nn.tanh, name='big')(input_layer)
middle = tf.keras.layers.Dense(200, activation=tf.nn.tanh, name='middle')(big_layer)
moveprobability = tf.keras.layers.Dense(cardamount*amountofbatches, activation=tf.nn.sigmoid, name='moveprobability')(middle)
winprobability = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='winprobability')(middle)
allowedcardprobability = tf.keras.layers.Multiply(name='finalmoveprobability')([allowed_moves, moveprobability])
enemycardprobability = tf.keras.layers.Dense(cardamount*amountofbatches, activation=tf.nn.sigmoid, name='enemyprobability')(middle)
enemycardprobabilitysoftmax = tf.keras.layers.Dense(cardamount*amountofbatches, activation=tf.nn.softmax, name='enemyprobabilitysoftmax')(enemycardprobability)

model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                            outputs=[allowedcardprobability, winprobability, enemycardprobabilitysoftmax])
model.compile(optimizer='Adam',
                   loss=[tf.keras.losses.categorical_crossentropy, tf.compat.v2.losses.mean_squared_error, tf.keras.losses.categorical_crossentropy],
                   metrics=['accuracy'])

game_history_starter = []
game_history_no_starter = []

class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, tfmodel):
        self.model = tfmodel

    def __len__(self):
        return 10001

    def __getitem__(self, item):
        if item % 100 == 0:
            model.save_weights("result2/model" + str(item))
            won_games = 0
            for test_number in range(0, 10):
                board = Board(cardshandedtoeachplayer, playeramount, cardamount, amountofbatches, maxbatchcards)
                players = []
                for player_number in range(0, playeramount):
                    if player_number == 0:
                        players.append(Player(model, player_number))
                    else:
                        players.append(TestPlayer(player_number))
                game_not_finished = True
                while game_not_finished:
                    selected_moves = []
                    for player_number in range(0, playeramount):
                        move_probability, player_input, allowed_move_input = players[player_number].calculate_turn(
                            board)
                        selected_move = int(np.random.choice(len(move_probability), p=move_probability))
                        selected_moves.append(selected_move)
                    game_not_finished = board.add_move(selected_moves)
                game_feedback = board.get_feedback_for_player(0)
                won_games = won_games + game_feedback
            game_history_starter.append(won_games)
            f = open("result2/starter.txt", "w")
            f.write(str(game_history_starter))
            f.close()
        board = Board(cardshandedtoeachplayer, playeramount, cardamount, amountofbatches, maxbatchcards)
        players = []
        training_input = []
        allowed_moves_input = []
        move_probabilities = []
        enemy_probabilities = []
        win_probabilities = []
        for player_number in range(0, playeramount):
            players.append(Player(model, player_number))
            training_input.append([])
            allowed_moves_input.append([])
            move_probabilities.append([])
            enemy_probabilities.append([])
        game_not_finished = True
        while game_not_finished:
            selected_moves = []
            for player_number in range(0, playeramount):
                move_probability, player_input, allowed_move_input = players[player_number].calculate_turn(board)
                training_input[player_number].append(player_input)
                allowed_moves_input[player_number].append(allowed_move_input)
                move_probabilities[player_number].append(move_probability)
                selected_move = int(np.random.choice(len(move_probability), p=move_probability))
                selected_moves.append(selected_move)
            for player_number in range(0, playeramount):
                enemy_probabilitiy = [0]*(cardamount*amountofbatches)
                for move_number in range(0, len(selected_moves)):
                    if move_number != player_number:
                        enemy_probabilitiy[selected_moves[move_number]] = 1/(len(selected_moves)-1)
                enemy_probabilities[player_number].append(enemy_probabilitiy)
            game_not_finished = board.add_move(selected_moves)
        for player_number in range(0, playeramount):
            game_feedback = board.get_feedback_for_player(player_number)
            win_probability = [game_feedback] * cardshandedtoeachplayer
            win_probabilities.append(win_probability)
        final_input = []
        final_allow = []
        final_move = []
        final_win = []
        final_enemy = []
        for player_number in range(0, playeramount):
            final_input.extend(training_input[player_number])
            final_allow.extend(allowed_moves_input[player_number])
            final_move.extend(move_probabilities[player_number])
            final_win.extend(win_probabilities[player_number])
            final_enemy.extend(enemy_probabilities[player_number])
        return {'input': np.array(final_input), 'allow': np.array(final_allow)}, {'finalmoveprobability': np.array(final_move), 'winprobability': np.array(final_win), 'enemyprobabilitysoftmax': np.array(final_enemy)}


generator = GameStateGenerator(model)

model.fit_generator(generator=generator, epochs=1, workers=0, max_queue_size=0, verbose=1, shuffle=False)
