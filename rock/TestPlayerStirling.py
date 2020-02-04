import requests

from nimmt.Board import Board


class TestPlayerStirling:
    def __init__(self, player_number: int):
        self.player_number = player_number
        url = 'http://www.cs.stir.ac.uk/~kms/schools/rps/index.php'
        x = requests.get(url)
        self.cookies = x.cookies._find('PHPSESSID')
        print(self.cookies)

    def calculate_turn(self, move: int):
        move_string = ''
        if move == 0:
            move_string = 'Rock'
        if move == 1:
            move_string = 'Paper'
        if move == 2:
            move_string = 'Scissors'
        url = 'http://www.cs.stir.ac.uk/~kms/schools/rps/index.php'
        myobj = {'action': move_string}
        x = requests.post(url, data=myobj, cookies={"PHPSESSID": self.cookies}, headers={'Content-Type': "application/x-www-form-urlencoded"})
        text = x.text
        index = text.find('I picked')
        pick = text[index:].split('.')[0][9:]
        return_value = -1
        print(pick)
        if pick == 'Rock':
            return_value = 0
        if pick == 'Paper':
            return_value = 1
        if pick == 'Scissors':
            return_value = 2
        print(return_value)
        return return_value, [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_player_number(self) -> int:
        return self.player_number
