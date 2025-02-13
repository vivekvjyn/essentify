from essentia.standard import KeyExtractor

class KeyEstimator:
    def estimate_key(self, audio, profileType):
        key, scale, strength = KeyExtractor(profileType=profileType)(audio)

        return key, scale
