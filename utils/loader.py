from essentia.standard import AudioLoader, MonoMixer, Resample

def load_audio(filename):
    audio, sampleRate, numChannels, md5, bit_rate, codec = AudioLoader(filename=filename)()
    
    audio_downmixed = MonoMixer()(audio, numChannels)

    audio_resampled = Resample(inputSampleRate=sampleRate, outputSampleRate=16000)(audio_downmixed)

    return audio, audio_downmixed, audio_resampled


