from essentia.standard import TempoCNN

class TempoEstimator:
    def __init__(self):
        self.model = TempoCNN(graphFilename="models/weights/deepsquare-k16-3.pb")
    
    def estimate_tempo(self, audio):
        global_tempo, local_tempo, local_tempo_probabilities = self.model(audio)

        return global_tempo
