import random
import threading

import numpy as np
import tensorflow as tf

from tictactoe.Board import Board
from tictactoe.Player import Player


class AITrainingProcess(threading.Thread):

    def __init__(self, ai_trainer, is_starter):
        self.ai_trainer = ai_trainer
        input = tf.keras.Input(shape=(18,), name='input')
        allowed_moves = tf.keras.Input(shape=(9,), name='allow')
        middle = tf.keras.layers.Dense(100, activation=tf.nn.tanh, name='middle')(input)
        moveprobability = tf.keras.layers.Dense(9, activation=tf.nn.softmax, name='moveprobability')(middle)
        winprobability = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='winprobability')(middle)
        allowedcardprobability = tf.keras.layers.Multiply()([allowed_moves, moveprobability])

        self.model = tf.keras.Model(inputs=[input, allowed_moves],
                               outputs=[allowedcardprobability, winprobability])
        self.model.compile(optimizer='adam',
                      loss=[tf.compat.v2.losses.sparse_categorical_crossentropy, tf.compat.v2.losses.mean_squared_error],
                      metrics=['accuracy'])
        self.model_is_starter = is_starter
        self.training_inputs = []
        self.allowed_moves = []
        self.move_probabilities = []
        self.win_probabilities = []

    def run(self) -> None:
        old_model_list = self.ai_trainer.getOldAI()
        while self.ai_trainer.continue_training:
            last_game_was_victory = True
            old_model_slice = old_model_list.copy()
            self.training_inputs = []
            self.allowed_moves = []
            self.move_probabilities = []
            self.win_probabilities = []
            while last_game_was_victory:
                random_old_model = old_model_slice[random.randint(0, len(old_model_slice))]
                last_game_was_victory = self.play_game(random_old_model)
                old_model_slice.remove(random_old_model)
                if last_game_was_victory:
                    if len(old_model_slice) == 0:
                        self.ai_trainer.continue_training = False
                        self.ai_trainer.old_ai.append(self.model)
                else:
                    self.model.train({'input': np.array(self.training_inputs), 'allow': np.array(self.allowed_moves)}, {'moveprobability': np.array(self.move_probabilities), 'winprobability': np.array(self.win_probabilities)})

    def play_game(self, ai_old):
        player_with_current_ai = Player(self.model, 0)
        player_with_old_ai = Player(ai_old, 1)
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
        while game_not_finished:
            if current_ai_plays:
                move_probability, player_input, allowed_moves = player_with_current_ai.calculate_turn(board)
                training_input_current_player.append(player_input)
                allowed_moves_current_player.append(allowed_moves)
                move_probabilities_current_player.append(move_probability)
                current_ai_plays = False
                player_number = player_with_current_ai.get_player_number()
            else:
                move_probability, player_input = player_with_old_ai.calculate_turn(board)
                training_input_old_player.append(player_input)
                allowed_moves_old_player.append(allowed_moves)
                move_probabilities_old_player.append(move_probability)
                current_ai_plays = True
                player_number = player_with_old_ai.get_player_number()
            move = np.argmax(move_probability)
            current_player_won, game_not_finished = board.add_move(player_number, move)
        if current_player_won:
            win_probabilities_current_player = [1] * len(training_input_current_player)
            win_probabilities_old_player = [-1] * len(training_input_old_player)
        else:
            win_probabilities_current_player = [-1] * len(training_input_current_player)
            win_probabilities_old_player = [1] * len(training_input_old_player)
        self.training_inputs.extend(training_input_current_player)
        self.training_inputs.extend(training_input_old_player)
        self.allowed_moves.extend(allowed_moves_current_player)
        self.allowed_moves.extend(allowed_moves_old_player)
        self.move_probabilities.extend(move_probabilities_current_player)
        self.move_probabilities.extend(move_probabilities_old_player)
        self.win_probabilities.extend(win_probabilities_current_player)
        self.win_probabilities.extend(win_probabilities_old_player)
        return current_player_won
