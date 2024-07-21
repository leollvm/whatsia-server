from TTS.api import TTS

# Carregamento do modelo xtts_v2 durante a inicialização
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Função para gerar áudio a partir de texto
def generate_audio(text, file_path, speaker_wav, language='en'):
    tts.tts_to_file(text=text, file_path=file_path, speaker_wav=speaker_wav, language=language)

# Exemplo de uso da função
generate_audio(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent. Quero comer Hamburguer",
               file_path="clonev1.wav",
               speaker_wav=["teste.wav"])
