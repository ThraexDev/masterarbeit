import numpy as np
import tensorflow as tf

from nimmt.Board import Board
from nimmt.Player import Player

cardshandedtoeachplayer = 10
playeramount = 2
cardamount = 51
amountofbatches = 2
maxbatchcards = 3

input_layer = tf.keras.Input(shape=(294,), name='input')
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
model.load_weights("result/model3600")

board = Board(cardshandedtoeachplayer, playeramount, cardamount, amountofbatches, maxbatchcards)
players = []
for player_number in range(0, playeramount):
    players.append(Player(model, player_number))
game_not_finished = True
while game_not_finished:
    selected_moves = []
    for player_number in range(0, playeramount):
        if player_number == 0:
            indices = [i for i, x in enumerate(board.playercards[0]) if x == 1]
            print('bulls:')
            for j in range(0, playeramount):
                print('Player '+str(j)+': '+str(board.playerbulls[j]))
            print('batch:')
            batch_number = 0
            for print_batch in board.batch:
                batch_number = batch_number + 1
                print(str(batch_number)+': '+str(print_batch)+' -> amount of bulls: '+ str(board.calculatebulls(print_batch)))
            print('hand:')
            print(indices)
            card = int(input('card that you want to play: '))
            batch = int(input('batch that you would take: '))
            selected_move = card + ((batch-1) * cardamount)
        else:
            move_probability, player_input, allowed_move_input = players[player_number].calculate_turn(
                board)
            selected_move = int(np.random.choice(len(move_probability), p=move_probability))
        selected_moves.append(selected_move)
    game_not_finished = board.add_move(selected_moves)

print('bulls:')
for j in range(0, playeramount):
    print('Player ' + str(j) + ': ' + str(board.playerbulls[j]))
print('batch:')
batch_number = 0
for print_batch in board.batch:
    batch_number = batch_number + 1
    print(
        str(batch_number) + ': ' + str(print_batch) + ' -> amount of bulls: ' + str(board.calculatebulls(print_batch)))

print("feedback: ")
print(board.get_feedback_for_player(0))

