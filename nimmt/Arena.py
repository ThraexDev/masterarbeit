import numpy as np
import tensorflow as tf

from nimmt.Board import Board
from nimmt.Player import Player

cardshandedtoeachplayer = 10
playeramount = 3
cardamount = 61
amountofbatches = 3
maxbatchcards = 5

def make_model(model_path: str):
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
    model.load_weights(model_path)
    return model

new_model = make_model("result4/model4300")
old_model = make_model("result4/model0")
won_games = 0
for test_number in range(0, 300):
    print("game: "+str(test_number))
    board = Board(cardshandedtoeachplayer, playeramount, cardamount, amountofbatches, maxbatchcards)
    players = []
    for player_number in range(0, playeramount):
        if player_number == 0:
            players.append(Player(new_model, player_number))
        else:
            players.append(Player(old_model, player_number))
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

print(won_games)
