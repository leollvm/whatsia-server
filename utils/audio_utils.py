import io
import numpy as np
from scipy.io.wavfile import write

def save_audio_to_wav(audio_np, sample_rate):
    audio_pcm = (audio_np * 32767).astype(np.int16)
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_pcm)
    buffer.seek(0)
    return buffer
