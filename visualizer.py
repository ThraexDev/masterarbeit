import matplotlib.pyplot as plt
f = open("tictactoe/resultsfinal/starter.txt", "r")
text = f.read()
historywon = list(map(float, text[1:-1].split(", ")))
N =1
cumsum, moving_aves_won = [0], []
for i, x in enumerate(historywon, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves_won.append(moving_ave)
x = list(range(0,len(moving_aves_won)))
new_x = [i * 100 for i in x]
plt.ylim(top=max(moving_aves_won))
plt.ylim(bottom=min(moving_aves_won))
plt.ylabel('Elo rating against baseline AI')
plt.xlabel('iteration')
zeroline = [0] * len(moving_aves_won)
plt.plot(new_x, moving_aves_won, color='green', label='Elo AlphaZero')
plt.plot(new_x, zeroline, color='blue', label='Elo comparison AI')
plt.legend(loc='upper left')
plt.show()