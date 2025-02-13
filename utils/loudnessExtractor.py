from essentia.standard import LoudnessEBUR128

class LoudnessExtractor:
    def __init__(self):
        self.loudnessEBUR128 = LoudnessEBUR128()

    def extract_loudness(self, audio):
        momentaryLoudness, shortTermLoudness, integratedLoudness, loudnessRange = self.loudnessEBUR128(audio)

        return integratedLoudness
