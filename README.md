# Master thesis - Comparing the viability of AlphaZero in games with perfect, imperfect, and no information
To create an AlphaZero model, go to the folder of the game and execute the Generator.py.
When the generator finished (this might take a while), it is possible to plot the Elo-rating during the training (visualizer.py and visualizer_rates.py), to play against the AI (PlayAgainstAI.py), or to generate statistics about the playstyle of AlphaZero (Arena.py).
Please note that every execution of the generator creates a different AI, which might have a different behavoiur and playstyle.  

In the following, a short description about all the files in the repository is given:

File | Description
------------ | -------------
visualizer.py | Used to create a plot of the Elo-rating
visualizer_rates.py | Used to plot the win rate, loss rate, and draw rate
## TicTacToe
File | Description
------------ | -------------
Board.py | Represents the board of TicTacToe, used by the generator
Generator.py | Creates an AlphaZero model and trains it
MCSTLeafNode.py | Represents the MCST of TicTacToe, used by the generator
MCSTNode.py | Represents the MCST of TicTacToe, used by the generator
MCSTNodeInterface.py | Represents the MCST of TicTacToe, used by the generator
MCSTRootNode.py | Represents the MCST of TicTacToe, used by the generator
PlayAgainstAI.py | Play against an AlphaZero model
Player.py | Represents the player of TicTacToe, used by the generator
TestPlayer.py | Comparison AI TicTacToe
TestPlayerRandom.py | Random AI TicTacToe
arena.py | Used to analyize the playstyle of AlphaZero
## 6Nimmt
File | Description
------------ | -------------
Board.py | Represents the board of Take 6!, used by the generator
Generator.py | Creates an AlphaZero model and trains it
MCSTLeafNode.py | Represents the MCST of Take 6!, used by the generator
MCSTNode.py | Represents the MCST of Take 6!, used by the generator
MCSTNodeInterface.py | Represents the MCST of Take 6!, used by the generator
MCSTRootNode.py | Represents the MCST of Take 6!, used by the generator
PlayAgainstAI.py | Play against an AlphaZero model
Player.py | Represents the player of Take 6!, used by the generator
TestPlayer.py | Random AI Take 6!
TestPlayer2.py | Comparison AI Take 6!
Arena.py | Used to analyize the playstyle of AlphaZero
## Rock Paper Scissors
File | Description
------------ | -------------
Board.py | Represents the board of Rock Paper Scissors, used by the generator
Generator.py | Creates an AlphaZero model and trains it	
Player.py	| Represents the player of Take 6!, used by the generator
TestPlayer.py	| Primitive Comparison AI
TestPlayerStirling.py	| Queries the Rock Paper Scissors AI of the sterling University
