from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN
import numpy as np

class EmbeddingsGenerator:
    def __init__(self):
        self.effnet_model = TensorflowPredictEffnetDiscogs(graphFilename="models/weights/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        self.musicnn_model = TensorflowPredictMusiCNN(graphFilename="models/weights/msd-musicnn-1.pb", output="model/dense/BiasAdd")

    def generate_embeddings(self, audio):
        effnet_embeddings = self.effnet_model(audio)
        musicnn_embeddings = self.musicnn_model(audio)

        return effnet_embeddings, musicnn_embeddings
