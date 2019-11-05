from tictactoe.AITrainingProcess import AITrainingProcess


class AITrainer:
    def __init__(self, parallel_processes):
        self.old_ai = []
        self.parallel_processes = parallel_processes
        self.continue_training = True

    def run_training_session(self):
        self.continue_training = True
        training_processes = []
        for training_process_number in range(0, self.parallel_processes):
            training_processes[training_process_number] = AITrainingProcess(self)
            training_processes[training_process_number].start()

    def getOldAI(self):
        return self.old_ai