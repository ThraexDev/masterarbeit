import time

from tictactoe.AITrainingProcess import AITrainingProcess
import matplotlib.pyplot as plt


class AITrainer:

    def __init__(self, parallel_processes, max_number_of_iterations):
        self.old_ai = []
        self.parallel_processes = parallel_processes
        self.training_processes = [0] * parallel_processes
        self.max_number_of_iterations = max_number_of_iterations
        self.starter_data = []
        self.not_starter_data = []

    def run_training_session(self):
        is_starter = True
        for iteration_number in range(0, self.max_number_of_iterations):
            print('iteration number')
            print(iteration_number)
            for training_process_number in range(0, self.parallel_processes):
                self.training_processes[training_process_number] = AITrainingProcess(self.old_ai, is_starter)
                self.training_processes[training_process_number].start()
            iteration_is_over = False
            while not iteration_is_over:
                for training_process_number in range(0, len(self.training_processes)):
                    if self.training_processes[training_process_number].is_finished():
                        iteration_is_over = True
                        finished_model = self.training_processes[training_process_number].get_model()
                        if is_starter:
                            name = 'starter_'
                        else:
                            name = 'no_starter_'
                        finished_model.save_weights(name+str(iteration_number))
                        self.old_ai.append(finished_model)
                        if is_starter:
                            self.starter_data.append(self.training_processes[training_process_number].test_games_won)
                            f = open("starter.txt", "w")
                            f.write(str(self.starter_data))
                            f.close()
                        else:
                            self.not_starter_data.append(self.training_processes[training_process_number].test_games_won)
                            f = open("no_starter.txt", "w")
                            f.write(str(self.not_starter_data))
                            f.close()
                        is_starter = not is_starter
                        break
                time.sleep(1)
        x = list(range(0, len(self.not_starter_data)))
        plt.ylim(top=max(self.not_starter_data))
        plt.ylim(bottom=min(self.not_starter_data))
        plt.ylabel('win rate against base line ai in %')
        plt.xlabel('iteration')
        plt.plot(x, self.not_starter_data, color='green', label='win rate')
        plt.legend(loc='upper left')
        plt.show()


trainer = AITrainer(1, 30)
trainer.run_training_session()


