from essentia.standard import TensorflowPredict2D
import numpy as np

class DancabilityClassifier:
    def __init__(self):
        self.model = TensorflowPredict2D(graphFilename="models/weights/danceability-discogs-effnet-1.pb", output="model/Softmax")

    def is_dancable(self, embeddings):
        predictions = self.model(embeddings) / 3.0

        return not np.argmax(predictions[0]).astype(bool), predictions[0, 0]
