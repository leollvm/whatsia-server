from TTS.api import TTS

class TTSModel:
    def __init__(self, model_name):
        self.tts = TTS(model_name=model_name).to("cuda")

    def synthesize(self, text, speaker_wav=None, language=None):
        if speaker_wav:
            return self.tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        return self.tts.tts(text=text)
