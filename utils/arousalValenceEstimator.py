from essentia.standard import TensorflowPredict2D
import numpy as np

class ArousalValenceEstimator:
    def __init__(self):
        self.model = TensorflowPredict2D(graphFilename="models/weights/emomusic-msd-musicnn-2.pb", output="model/Identity")

    def estimate_arousal_and_valence(self, embeddings):
        predictions = self.model(embeddings)

        predictions_mean = np.mean(predictions, axis=0)

        arousal, valence = predictions_mean

        return arousal, valence
