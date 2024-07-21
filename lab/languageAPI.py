import os
import tempfile
from TTS.api import TTS
from langchain_community.llms import Ollama
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
from scipy.io.wavfile import write
import io
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import aiofiles
import whisper

# Configura o modelo do Ollama
cached_llm = Ollama(model="llama3", base_url="http://localhost:11434")

# FastAPI app
app = FastAPI()

# Configura o Whisper ASR
Modelwhisper = whisper.load_model("small")

# Configura o TTS padrão
default_tts_model = "tts_models/en/ljspeech/vits--neon"
tts_models = {
    '1': "tts_models/en/ljspeech/vits--neon",
    '2': "tts_models/multilingual/multi-dataset/xtts_v2",
    # Adicione mais modelos conforme necessário para outros códigos de imagem
}

# Configura o Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Inicialização do modelo TTS xtts_v2
tts_xtts_v2 = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def save_audio_to_wav(audio_np, sample_rate):
    # Ajusta a amplitude para o formato PCM de 16 bits (convenção: -32768 a +32767)
    audio_pcm = (audio_np * 32767).astype(np.int16)
    
    # Cria um buffer de bytes para armazenar o áudio em formato WAV
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_pcm)
    buffer.seek(0)
    return buffer

@app.post("/llm")
async def llmpost(session_id: str = Form(...), audio_file: UploadFile = File(...), image_code: str = Form(...)):
    print("Post /llm called")

    # Lê o arquivo de áudio
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_file_path = tmp.name
            async with aiofiles.open(temp_file_path, 'wb') as out_file:
                content = await audio_file.read()  # Ler o conteúdo do arquivo enviado
                await out_file.write(content)     # Escrever o conteúdo no arquivo temporário

        # Transcreve o áudio usando o Whisper ASR
        try:
            transcription_result = Modelwhisper.transcribe(temp_file_path)
        except Exception as e:
            return {"error": f"Failed to transcribe audio: {str(e)}"}, 500

        # Extrai apenas o texto da fala
        speech_text = transcription_result['text']

        # Armazena a transcrição na variável "query" no contexto da sessão
        if not session_id:
            return {"error": "Missing 'session_id' field in form data"}, 400

        conversation_context = redis_client.get(session_id)
        if conversation_context:
            conversation_context = json.loads(conversation_context)
        else:
            conversation_context = [
                "These are your premises: follow them, only respond to what is said after your premises"
                + " limit your answer to 100 characters maximum,"
                + " you are an English language teacher, your mission is to help me learn more, correct my mistakes in a friendly way,"
                + " also correct me if I conjugate the wrong verb, help me to be fluent in English"
                + " Don't repeat my speech, just go straight to the answer,"
                + " help me by encouraging me to talk to you more,"
                + " If my sentence is strange, try to understand the context of our conversation and ask if something similar to the words I said is actually correct,"
                + " If you create a list, do not number it or use symbols, instead, add commas and skip 2 lines"
                + " put a small pause when there are line breaks: (end of premises) "
            ]

        conversation_context.append(speech_text)
        redis_client.set(session_id, json.dumps(conversation_context))

        # Simula o processamento do texto usando o modelo Ollama
        full_context = " ".join([str(item) for item in conversation_context if isinstance(item, str)])
        response = cached_llm.invoke(full_context)

        print("Ollama response:", response)
        print(speech_text)
        print(transcription_result)

        # Adiciona a resposta do chatbot ao contexto da conversa
        conversation_context.append(response)

        # Salva o contexto atualizado no Redis
        redis_client.set(session_id, json.dumps(conversation_context))

        # Seleciona o modelo TTS com base no código da imagem recebido
        tts_model = tts_models.get(image_code, default_tts_model)
        tts = TTS(model_name=tts_model)

        # Gera a resposta em áudio usando o TTS selecionado
        if tts_model == "tts_models/multilingual/multi-dataset/xtts_v2":
            audio_response = tts_xtts_v2.tts(text=response, speaker_wav=["teste.wav"], language='en')
        else:
            audio_response = tts.tts(text=response)

        if isinstance(audio_response, list):
            # Se audio_response for uma lista de bytes, tenta concatená-la
            try:
                audio_response_bytes = b"".join([bytes(chunk) for chunk in audio_response])
            except Exception as e:
                return {"error": f"Failed to concatenate audio bytes: {str(e)}"}, 500
        elif isinstance(audio_response, bytes):
            audio_response_bytes = audio_response
        else:
            return {"error": "Unexpected audio response type"}, 500

        try:
            # Converte o áudio para o formato WAV e armazena no buffer de memória
            audio_buffer = save_audio_to_wav(np.frombuffer(audio_response_bytes, dtype=np.float32), tts.synthesizer.output_sample_rate)

            # Envia o buffer de áudio como resposta HTTP com streaming
            return StreamingResponse(audio_buffer, media_type="audio/wav")

        except Exception as e:
            return {"error": f"Failed to process audio segment: {str(e)}"}, 500

    finally:
        # Remove o arquivo temporário
        if temp_file_path:
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
