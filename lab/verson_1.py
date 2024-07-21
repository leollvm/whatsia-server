import io
import numpy as np
from TTS.api import TTS
from langchain_community.llms import Ollama
from pydub import AudioSegment
from pydub.playback import play
from flask import Flask, request

# Configura o modelo do Ollama
cached_llm = Ollama(model="llama3", base_url="http://localhost:11434")

#Flask API
app = Flask(__name__)

@app.route("/llm", methods=["POST"])
def llmpost():
    print("Post /llm called")
    json_context = request.json
    # Pessoa
    query = json_context.get("query")


    # IA    
    response = cached_llm.invoke(
            "These are your premises: follow them, only respond to what is said after your premises"
        + " limit your answer to 100 characters maximum,"
        + " you are a English language teacher, your mission is to help me learn more, correct my mistakes in a friendly way,"
        + " also correct me if I conjugate the wrong verb, help me to be fluent in English"
        + " Don't repeat my speech, just go straight to the answer,"
        + " help me by encouraging me to talk to you more"
        + " If you create a list, do not number it or use symbols, instead, add commas and skip 2 lines"
        + " put a small pause when there are line breaks: (end of premises) " + query
    )

    print(response)

    # Configura o TTS
    tts = TTS(model_name="tts_models/en/ljspeech/vits--neon")  #tts_models/en/ljspeech/vits--neon

    # Converte o texto em áudio usando a função tts
    wav = tts.tts(text=response)

    response_answer = {"answer": response}

    return response_answer

    # Definindo parâmetros para processamento por chunks
    chunk_size = 200000  # Tamanho do chunk em amostras (ajustado)
    audio_chunks = [wav[i:i+chunk_size] for i in range(0, len(wav), chunk_size)]

    # Processa e reproduz cada chunk
    for chunk in audio_chunks:
        # Converte o chunk de lista para numpy array
        chunk_np = np.array(chunk)
        
        # Converte o áudio do chunk para formato que pydub pode processar (16-bit PCM)
        audio_bytes = (chunk_np * 32767).astype(np.int16).tobytes()
        
        # Cria um objeto BytesIO com os bytes do áudio
        audio_stream = io.BytesIO(audio_bytes)
        
        # Cria um objeto AudioSegment a partir do BytesIO
        audio_segment = AudioSegment.from_file(audio_stream, format="raw", sample_width=2, channels=1, frame_rate=tts.synthesizer.output_sample_rate)
        
        # Reproduz o áudio do chunk
        play(audio_segment)


def start_app():
    app.run(host="0.0.0.0", port = 8080, debug=True)

if __name__ == "__main__":
    start_app()