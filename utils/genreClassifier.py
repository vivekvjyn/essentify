from essentia.standard import TensorflowPredict2D
import json
import numpy as np

class GenreClassifier:
    def __init__(self):
        self.model = TensorflowPredict2D(graphFilename="models/weights/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
        
        with open("models/metadata/genre_discogs400-discogs-effnet-1.json", "r") as f:
            self.mappings = json.load(f)["classes"]

    def classify_genre(self, embeddings):
        predictions = self.model(embeddings)

        predictions_mean = np.mean(predictions, axis=0)

        top_prediction = np.argmax(predictions_mean)

        predicted_class = self.mappings[top_prediction]

        parent_class, child_class = predicted_class.split('---')

        return parent_class, child_class, predictions_mean 
