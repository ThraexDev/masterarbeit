import random
import threading
import time

import numpy as np
import tensorflow as tf

from tictactoe.Board import Board
from tictactoe.Player import Player

class AITrainingProcess(threading.Thread):

    def __init__(self, old_model_list, is_starter: bool):
        super().__init__()
        input_layer = tf.keras.Input(shape=(18,), name='input')
        allowed_moves = tf.keras.Input(shape=(9,), name='allow')
        middle = tf.keras.layers.Dense(100, activation=tf.nn.tanh, name='middle')(input_layer)
        moveprobability = tf.keras.layers.Dense(9, activation=tf.nn.softmax, name='moveprobability')(middle)
        winprobability = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='winprobability')(middle)
        allowedcardprobability = tf.keras.layers.Multiply()([allowed_moves, moveprobability])

        self.model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                               outputs=[allowedcardprobability, winprobability])
        self.model.compile(optimizer='adam',
                      loss=[tf.compat.v2.losses.sparse_categorical_crossentropy, tf.compat.v2.losses.mean_squared_error],
                      metrics=['accuracy'])
        self.model_is_starter = is_starter
        self.training_inputs = []
        self.allowed_moves = []
        self.move_probabilities = []
        self.win_probabilities = []
        self.thread_finished = False
        self.old_model_list = old_model_list

    def is_finished(self):
        return self.thread_finished

    def get_model(self):
        return self.model

    def run(self) -> None:
        old_model_list = self.old_model_list
        if len(old_model_list) == 0:
            self.thread_finished = True
        while not self.thread_finished:
            print("iteration")
            last_game_was_victory = True
            old_model_slice = old_model_list.copy()
            self.training_inputs = []
            self.allowed_moves = []
            self.move_probabilities = []
            self.win_probabilities = []
            while last_game_was_victory:
                print("last game victory")
                if len(old_model_slice) > 1:
                    random_int = random.randint(0, len(old_model_slice)-1)
                else:
                    random_int = 0
                print(len(old_model_slice))
                print(random_int)
                random_old_model = old_model_slice[random_int]
                last_game_was_victory = self.play_game(random_old_model)
                old_model_slice.remove(random_old_model)
                if last_game_was_victory:
                    if len(old_model_slice) == 0:
                        print("iteration end")
                        self.thread_finished = True
                else:
                    self.model.train({'input': np.array(self.training_inputs), 'allow': np.array(self.allowed_moves)}, {'moveprobability': np.array(self.move_probabilities), 'winprobability': np.array(self.win_probabilities)})

    def play_game(self, ai_old: tf.keras.Model):
        player_with_current_ai = Player(self.model, 0)
        player_with_old_ai = Player(ai_old, 1)
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


class AITrainer:

    def __init__(self, parallel_processes, max_number_of_iterations):
        self.old_ai = []
        self.parallel_processes = parallel_processes
        self.training_processes = [0] * parallel_processes
        self.max_number_of_iterations = max_number_of_iterations

    def run_training_session(self, is_starter: bool):
        for iteration_number in range(0, self.max_number_of_iterations):
            for training_process_number in range(0, self.parallel_processes):
                print("start " + str(training_process_number))
                self.training_processes[training_process_number] = AITrainingProcess(self.old_ai, is_starter)
                self.training_processes[training_process_number].start()
            iteration_is_over = False
            while not iteration_is_over:
                for training_process_number in range(0, len(self.training_processes)):
                    if self.training_processes[training_process_number].is_finished():
                        print("finished")
                        iteration_is_over = True
                        self.old_ai.append(self.training_processes[training_process_number].get_model())
                        break
                time.sleep(1)


trainer = AITrainer(4, 10)
trainer.run_training_session(True)


