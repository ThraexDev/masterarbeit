import numpy as np
import tensorflow as tf

from nimmt.Board import Board
from nimmt.Player import Player
from nimmt.TestPlayer import TestPlayer
from nimmt.TestPlayer2 import TestPlayer2

cardshandedtoeachplayer = 10
playeramount = 3
cardamount = 104
amountofbatches = 3
maxbatchcards = 5

def make_model(model_path: str):
    input_layer = tf.keras.Input(shape=(650,), name='input')
    allowed_moves = tf.keras.Input(shape=(cardamount * amountofbatches,), name='allow')
    big_layer = tf.keras.layers.Dense(800, activation=tf.nn.tanh, name='big')(input_layer)
    middle = tf.keras.layers.Dense(200, activation=tf.nn.tanh, name='middle')(big_layer)
    moveprobability = tf.keras.layers.Dense(cardamount * amountofbatches, activation=tf.nn.sigmoid,
                                            name='moveprobability')(middle)
    winprobability = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='winprobability')(middle)
    allowedcardprobability = tf.keras.layers.Multiply(name='finalmoveprobability')([allowed_moves, moveprobability])
    enemycardprobability = tf.keras.layers.Dense(cardamount * amountofbatches, activation=tf.nn.sigmoid,
                                                 name='enemyprobability')(middle)
    enemycardprobabilitysoftmax = tf.keras.layers.Dense(cardamount * amountofbatches, activation=tf.nn.softmax,
                                                        name='enemyprobabilitysoftmax')(enemycardprobability)

    model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                           outputs=[allowedcardprobability, winprobability, enemycardprobabilitysoftmax])
    model.compile(optimizer='Adam',
                  loss=[tf.keras.losses.categorical_crossentropy, tf.compat.v2.losses.mean_squared_error,
                        tf.keras.losses.categorical_crossentropy],
                  metrics=['accuracy'])
    model.load_weights(model_path)
    return model

new_model = make_model("resultfinal/model10000")
old_model = make_model("resultfinal/model10000")
won_games = 0
lost_games = 0
draw_games = 0
recorded_moves = []
recorded_bulls = []
recorded_batch_length = []
for i in range(0, cardshandedtoeachplayer):
    recorded_moves.append([])
    recorded_bulls.append([])
    recorded_batch_length.append([])
for test_number in range(0, 1000):
    print("game: "+str(test_number))
    board = Board(cardshandedtoeachplayer, playeramount, cardamount, amountofbatches, maxbatchcards)
    players = []
    for player_number in range(0, playeramount):
        if player_number == 0:
            players.append(TestPlayer2(player_number))
            players.append(Player(new_model, player_number))
        else:
            players.append(Player(new_model, player_number))
            players.append(TestPlayer(player_number))
    game_not_finished = True
    move_number = 0
    while game_not_finished:
        selected_moves = []
        bull_sum = 0
        batch_length = 0
        for player_number in range(0, playeramount):
            move_probability, player_input, allowed_move_input = players[player_number].calculate_turn(
                board)
            selected_move = int(np.random.choice(len(move_probability), p=move_probability))
            selected_moves.append(selected_move)
            recorded_moves[move_number].append(selected_move % cardamount)
        game_not_finished = board.add_move(selected_moves)
        for player_number in range(0, playeramount):
            bull_sum = bull_sum + board.playerbulls[player_number]
        recorded_bulls[move_number].append(bull_sum)
        for i in range(0, amountofbatches):
            batch_length = batch_length + len(board.batch[i])
        recorded_batch_length[move_number].append(batch_length)
        move_number = move_number + 1
    game_feedback = board.get_feedback_for_player(0)
    if game_feedback == 1:
        won_games = won_games + 1
    if game_feedback == 0:
        draw_games = draw_games +1
    if game_feedback == -1:
        lost_games = lost_games + 1

print(won_games)
print(draw_games)
print(lost_games)
print('--------')
for recorded_move in recorded_moves:
    print('average number on card played: '+str(sum(recorded_move)/(100*playeramount)))
print('--------')
for recorded_bull in recorded_bulls:
    print('average bulls per player: '+str(sum(recorded_bull)/(100*playeramount)))
print('--------')
for recorded_batch in recorded_batch_length:
    print('average batch length: '+str(sum(recorded_batch)/(100*amountofbatches)))
