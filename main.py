import os
import json
from tqdm import tqdm
import essentia
import numpy as np

essentia.log.infoActive = False
essentia.log.warningActive = False

from utils.loader import load_audio
from utils.tempoEstimator import TempoEstimator
from utils.keyEstimator import KeyEstimator
from utils.loudnessExtractor import LoudnessExtractor
from utils.embeddingsGenerator import EmbeddingsGenerator
from utils.genreClassifier import GenreClassifier
from utils.instrumentalClassifier import InstrumentalClassifier
from utils.danceabilityClassifier import DanceabilityClassifier
from utils.arousalValenceEstimator import ArousalValenceEstimator

PATH = './audio'
SAVE_PATH = './results'
num_files = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(PATH)])

def analyze_files(pbar, path=PATH):
    tempoEstimator = TempoEstimator()
    keyEstimator = KeyEstimator()
    loudnessExtractor = LoudnessExtractor()
    embeddingsGenerator = EmbeddingsGenerator()
    genreClassifier = GenreClassifier()
    instrumentalClassifier = InstrumentalClassifier()
    danceabilityClassifier = DanceabilityClassifier()
    arousalValenceEstimator = ArousalValenceEstimator()

    results = {}
    embeddings = {}
    activations = {}

    for dirpath, dirnames, filenames in os.walk(path):

        for filename in filenames:
            audio, audio_downmixed, audio_resampled = load_audio(os.path.join(dirpath, filename))

            tempo = tempoEstimator.estimate_tempo(audio_resampled)

            temperley_key, temperley_scale = keyEstimator.estimate_key(audio_downmixed, 'temperley')
            krumhansl_key, krumhansl_scale = keyEstimator.estimate_key(audio_downmixed, 'krumhansl')
            edma_key, edma_scale = keyEstimator.estimate_key(audio_downmixed, 'edma')

            loudness = loudnessExtractor.extract_loudness(audio)

            effnet_embeddings, musicnn_embeddings = embeddingsGenerator.generate_embeddings(audio_resampled)

            genre, sub_genre, genre_activations = genreClassifier.classify_genre(effnet_embeddings)

            is_instrumental = instrumentalClassifier.is_instrumental(effnet_embeddings)

            is_danceable, confidence = danceabilityClassifier.is_dancable(effnet_embeddings)

            arousal, valence = arousalValenceEstimator.estimate_arousal_and_valence(musicnn_embeddings)
                
            results.update({
                os.path.join(dirpath, filename): {
                    'tempo': tempo,
                    'key': (temperley_key, krumhansl_key, edma_key),
                    'key (edma)': edma_key,
                    'scale': (temperley_scale, krumhansl_scale, edma_scale),
                    'scale (edma)': edma_scale,
                    'loudness': loudness,
                    'genre': genre,
                    'sub genre': sub_genre,
                    'instrumental': 'Instrumental' if is_instrumental else 'Voice',
                    'danceability': 'Danceable' if is_danceable else 'Not danceable',
                    'danceability confidence': confidence.astype(float),
                    'arousal': arousal.astype(float),
                    'valence': valence.astype(float)
                }
            })

            embeddings.update({
                os.path.join(dirpath, filename): {
                    'effnet_embeddings': np.mean(effnet_embeddings, axis=0).tolist(),
                    'musicnn_embeddings': np.mean(musicnn_embeddings, axis=0).tolist()
                }
            })

            activations.update({
                os.path.join(dirpath, filename): genre_activations.tolist()
            })

            pbar.update(1)

    return results, embeddings, activations


def save_results(results, embeddings, activations, save_path=SAVE_PATH):
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(save_path, 'embeddings.json'), 'w') as f:
        json.dump(embeddings, f, indent=4)

    with open(os.path.join(save_path, 'activations.json'), 'w') as f:
        json.dump(activations, f, indent=4)


def main():
    with tqdm(desc="Processing files", total=num_files, unit="file") as pbar:
        results, embeddings, activations = analyze_files(pbar)

    save_results(results, embeddings, activations)
        
if __name__ == '__main__':
    main()
