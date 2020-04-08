class Nn:
    def __init__(self, input, hidden, output):
        self.input_neurons = input
        self.hidden_neurons = hidden
        self.output_neurons = output
        print("created Neural network")

    def inputNeurons(self):
        return self.input_neurons
