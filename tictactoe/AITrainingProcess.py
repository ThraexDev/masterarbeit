import random
import threading

import numpy as np
import tensorflow as tf

from tictactoe.Board import Board
from tictactoe.Player import Player
from tictactoe.TestPlayer import TestPlayer


class AITrainingProcess(threading.Thread):

    def __init__(self, old_model_list, is_starter: bool):
        super().__init__()
        input_layer = tf.keras.Input(shape=(18,), name='input')
        allowed_moves = tf.keras.Input(shape=(9,), name='allow')
        middle = tf.keras.layers.Dense(100, activation=tf.nn.tanh, name='middle')(input_layer)
        moveprobability = tf.keras.layers.Dense(9, activation=tf.nn.sigmoid, name='moveprobability')(middle)
        winprobability = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='winprobability')(middle)
        allowedcardprobability = tf.keras.layers.Multiply(name='finalmoveprobability')([allowed_moves, moveprobability])

        self.model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                               outputs=[allowedcardprobability, winprobability])
        self.model.compile(optimizer='adam',
                      loss=[tf.compat.v2.losses.CategoricalCrossentropy(), tf.compat.v2.losses.mean_squared_error],
                      metrics=['accuracy'])
        self.model_is_starter = is_starter
        self.training_inputs = []
        self.allowed_moves = []
        self.move_probabilities = []
        self.win_probabilities = []
        self.thread_finished = False
        self.old_model_list = old_model_list
        self.test_games_won = 0

    def is_finished(self):
        return self.thread_finished

    def get_model(self):
        return self.model

    def run(self) -> None:
        old_model_list = self.old_model_list
        if len(old_model_list) == 0:
            self.thread_finished = True
        while not self.thread_finished:
            last_game_was_victory = True
            old_model_slice = old_model_list.copy()
            self.training_inputs = []
            self.allowed_moves = []
            self.move_probabilities = []
            self.win_probabilities = []
            while last_game_was_victory:
                if len(old_model_slice) > 1:
                    random_int = random.randint(0, len(old_model_slice)-1)
                else:
                    random_int = 0
                random_old_model = old_model_slice[random_int]
                if self.model_is_starter:
                    player_number = 1
                else:
                    player_number = 0
                last_game_was_victory = self.play_game(Player(random_old_model, player_number))
                print(str(len(old_model_slice))+ '/'+ str(len(old_model_list)))
                if last_game_was_victory:
                    print("won")
                    old_model_slice.remove(random_old_model)
                    if len(old_model_slice) == 0:
                        test_games_won = 0
                        for test_game_number in range(0, 100):
                            if self.play_game(TestPlayer(player_number)):
                                test_games_won = test_games_won + 1
                        self.test_games_won = test_games_won
                        self.thread_finished = True
                        break
                else:
                    print("lost")
                    if self.model_is_starter: print('starter')
                    print(random_int)
                    print(self.training_inputs)
                    print(self.allowed_moves)
                    print(self.move_probabilities)
                    print(self.win_probabilities)
                    self.model.fit({'input': np.array(self.training_inputs), 'allow': np.array(self.allowed_moves)}, {'finalmoveprobability': np.array(self.move_probabilities), 'winprobability': np.array(self.win_probabilities)})

    def play_game(self, player_with_old_ai: Player):
        board: Board
        board = Board()
        training_input_current_player = []
        training_input_old_player = []
        allowed_moves_current_player = []
        allowed_moves_old_player = []
        move_probabilities_current_player = []
        move_probabilities_old_player = []
        win_probabilities_current_player = []
        win_probabilities_old_player = []
        game_not_finished = True
        current_ai_plays = False
        current_player_won = False
        if self.model_is_starter:
            current_ai_plays = True
            player_with_current_ai = Player(self.model, 0)
        else:
            player_with_current_ai = Player(self.model, 1)
        while game_not_finished:
            if current_ai_plays:
                move_probability, player_input, allowed_moves = player_with_current_ai.calculate_turn(board)
                training_input_current_player.append(player_input)
                allowed_moves_current_player.append(allowed_moves)
                move_probabilities_current_player.append(move_probability)
                current_ai_plays = False
                player_number = player_with_current_ai.get_player_number()
            else:
                move_probability, player_input, allowed_moves = player_with_old_ai.calculate_turn(board)
                training_input_old_player.append(player_input)
                allowed_moves_old_player.append(allowed_moves)
                move_probabilities_old_player.append(move_probability)
                current_ai_plays = True
                player_number = player_with_old_ai.get_player_number()
            move = np.argmax(move_probability)
            current_player_won, game_not_finished = board.add_move(player_number, move)
        if (current_player_won and not current_ai_plays) or (not current_player_won and current_ai_plays):
            win_probabilities_current_player = [[1]] * len(training_input_current_player)
            win_probabilities_old_player = [[0]] * len(training_input_old_player)
            move_probabilities_old_player = [[0.0001]*9] * len(training_input_old_player)
            game_won = True
        else:
            win_probabilities_current_player = [[0]] * len(training_input_current_player)
            move_probabilities_current_player = [[0.0001]*9] * len(training_input_current_player)
            win_probabilities_old_player = [[1]] * len(training_input_old_player)
            game_won = False
        self.training_inputs.extend(training_input_current_player)
        self.training_inputs.extend(training_input_old_player)
        self.allowed_moves.extend(allowed_moves_current_player)
        self.allowed_moves.extend(allowed_moves_old_player)
        self.move_probabilities.extend(move_probabilities_current_player)
        self.move_probabilities.extend(move_probabilities_old_player)
        self.win_probabilities.extend(win_probabilities_current_player)
        self.win_probabilities.extend(win_probabilities_old_player)
        return game_won
