from essentia.standard import TensorflowPredict2D
import numpy as np

class InstrumentalClassifier:
    def __init__(self):
        self.model = TensorflowPredict2D(graphFilename="models/weights/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")

    def is_instrumental(self, embeddings):
        predictions = self.model(embeddings)

        return not np.argmax(predictions).astype(bool)
