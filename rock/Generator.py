import tensorflow as tf
import numpy as np

from rock.Board import Board
from rock.Player import Player
from rock.TestPlayer import TestPlayer
from rock.TestPlayerStirling import TestPlayerStirling

print("Num GPUs Available: ", tf.config.experimental.list_physical_devices())
print(tf.test.is_gpu_available())

input_layer = tf.keras.Input(shape=(300,), name='input')
middle = tf.keras.layers.Dense(200, activation=tf.nn.tanh, name='middle')(input_layer)
moveprobability = tf.keras.layers.Dense(3, activation=tf.nn.softmax, name='moveprobability')(middle)

model = tf.keras.Model(inputs=[input_layer],
                            outputs=[moveprobability])
model.compile(optimizer='Adam',
                   loss=[tf.keras.losses.categorical_crossentropy],
                   metrics=['accuracy'])

game_history_starter = []
game_history_win = []
game_history_loss = []
game_history_draw = []
game_history_stirling = []

class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, tfmodel):
        self.model = tfmodel

    def __len__(self):
        return 10001

    def __getitem__(self, item):
        if item % 100 == 0:
            model.save_weights("result/model" + str(item))
            board = Board()
            players = []
            players.append(Player(model, 0))
            players.append(TestPlayer(1))
            for game_number in range(0, 100):
                selected_moves = []
                for player_number in range(0, 2):
                    move_probability, player_input = players[player_number].calculate_turn(
                        board)
                    selected_move = int(np.random.choice(len(move_probability), p=move_probability))
                    selected_moves.append(selected_move)
                board.add_move(selected_moves)
            game_feedback = board.get_feedback_for_player(0)
            game_history_starter.append(game_feedback)
            f = open("result/starter.txt", "w")
            f.write(str(game_history_starter))
            f.close()

            board = Board()
            players = []
            players.append(Player(model, 0))
            players.append(TestPlayerStirling(1))
            for game_number in range(0, 100):
                selected_moves = []
                move_probability, player_input = players[0].calculate_turn(
                    board)
                print(move_probability)
                selected_move = int(np.random.choice(len(move_probability), p=move_probability))
                selected_moves.append(selected_move)
                sterling_move, player_input = players[1].calculate_turn(
                    selected_move)
                selected_moves.append(sterling_move)
                board.add_move(selected_moves)
            game_feedback = board.get_feedback_for_player(0)
            wins = board.get_feedback_win()
            losses = board.get_feedback_loss()
            draws = board.get_feedback_draw()
            game_history_stirling.append(game_feedback)
            game_history_win.append(wins)
            game_history_loss.append(losses)
            game_history_draw.append(draws)
            f = open("result/stirling.txt", "w")
            f.write(str(game_history_stirling))
            f.close()
            f = open("result/wins.txt", "w")
            f.write(str(game_history_win))
            f.close()
            f = open("result/loss.txt", "w")
            f.write(str(game_history_loss))
            f.close()
            f = open("result/draw.txt", "w")
            f.write(str(game_history_draw))
            f.close()

        board = Board()
        players = []
        training_input = []
        move_probabilities = []
        players.append(Player(model, 0))
        players.append(Player(model, 1))
        training_input.append([])
        training_input.append([])
        move_probabilities.append([])
        move_probabilities.append([])
        for game_number in range(0, 100):
            selected_moves = []
            for player_number in range(0, 2):
                move_probability, player_input = players[player_number].calculate_turn(board)
                training_input[player_number].append(player_input)
                selected_move = int(np.random.choice(len(move_probability), p=move_probability))
                selected_moves.append(selected_move)
            board.add_move(selected_moves)
            move_probabilities[0].append(board.get_correct_move_for(0))
            move_probabilities[1].append(board.get_correct_move_for(1))
        final_input = []
        final_move = []
        for player_number in range(0, 2):
            final_input.extend(training_input[player_number])
            final_move.extend(move_probabilities[player_number])
        return {'input': np.array(final_input)}, {'moveprobability': np.array(final_move)}


generator = GameStateGenerator(model)

model.fit_generator(generator=generator, epochs=1, workers=0, max_queue_size=0, verbose=1, shuffle=False)
