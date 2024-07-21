import os
import tempfile
import json
import aiofiles
import redis
from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from models.asr_model import ASRModel
from models.llm_model import LLMModel
from models.tts_model import TTSModel
from utils.audio_utils import save_audio_to_wav
import numpy as np

router = APIRouter()

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

asr_model = ASRModel()
llm_model = LLMModel()
tts_models = {
    '1': TTSModel("tts_models/multilingual/multi-dataset/xtts_v2"), #"tts_models/en/ljspeech/vits--neon"
    # Adicione mais modelos conforme necessário para outros códigos de imagem
}

SESSION_TTL = 900  # TTL em segundos (15 minutos)

def set_session(session_id: str, data: dict):
    redis_client.set(session_id, json.dumps(data))
    redis_client.expire(session_id, SESSION_TTL)

def get_session(session_id: str):
    session_data = redis_client.get(session_id)
    if session_data:
        redis_client.expire(session_id, SESSION_TTL)  # Renova o TTL
        return json.loads(session_data)
    return None

@router.post("/llm")
async def llmpost(session_id: str = Form(...), audio_file: UploadFile = File(...), image_code: str = Form(...), language_code: str = Form(...)):
    print("Post /llm called")

    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_file_path = tmp.name
            async with aiofiles.open(temp_file_path, 'wb') as out_file:
                content = await audio_file.read()
                await out_file.write(content)

        try:
            transcription_result = asr_model.transcribe(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

        speech_text = transcription_result['text']

        if language_code == 'en':
            language = 'English'
        elif language_code == 'es':
            language = 'Spanish'
        elif language_code == 'pt':
            language = 'Portuguese'
        else:
            language = 'English'

        if image_code == '1':
            nameGuardian = 'Sakura'
        elif image_code == '2':
            nameGuardian = 'Mio'
        else:
            nameGuardian = 'Sakura'

        if image_code == '1':
            genderGuardian = 'Feminine'
        elif image_code == '2':
            genderGuardian = 'Male'
        else:
            genderGuardian = 'Feminine'

        if not session_id:
            raise HTTPException(status_code=400, detail="Missing 'session_id' field in form data")

        conversation_context = get_session(session_id)
        if not conversation_context:
            conversation_context = [
                " These are your premises: follow them, only respond to what is said after your premises"
                + " limit your answer to 100 characters maximum,"
                + " you are an " + language + " language teacher, your mission is to help me learn more, correct my mistakes in a friendly way,"
                + " your name is " + nameGuardian + ', ' + genderGuardian
                + " also correct me if I conjugate the wrong verb, help me to be fluent in " + language
                + " Don't repeat my speech, just go straight to the answer,"
                + " help me by encouraging me to talk to you more,"
                + " If my sentence is strange, try to understand the context of our conversation and ask if something similar to the words I said is actually correct,"
                + " If you create a list, do not number it or use symbols, instead, add commas and skip 2 lines"
                + " put a small pause when there are line breaks: "
                + " you must respond in " + language + " (end of premises) "
            ]

        conversation_context.append(speech_text)
        set_session(session_id, conversation_context)

        full_context = " ".join([str(item) for item in conversation_context if isinstance(item, str)])
        response = llm_model.invoke(full_context)

        print("Ollama response:", response)
        print(speech_text)
        print(transcription_result)

        conversation_context.append(response)
        set_session(session_id, conversation_context)

        tts_model = tts_models.get(image_code, tts_models['1'])

        if image_code == '1':
            audio_response = tts_model.synthesize(text=response, speaker_wav=["gato_cinza_alfa.wav"], language=language_code)
        elif image_code == '2':
            audio_response = tts_model.synthesize(text=response, speaker_wav=["gato_laranja_alfa.wav"], language=language_code)
        else:
            audio_response = tts_model.synthesize(text=response, language=language_code)

        if isinstance(audio_response, list):
            try:
                audio_response_bytes = b"".join([bytes(chunk) for chunk in audio_response])
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to concatenate audio bytes: {str(e)}")
        elif isinstance(audio_response, bytes):
            audio_response_bytes = audio_response
        else:
            raise HTTPException(status_code=500, detail="Unexpected audio response type")

        try:
            audio_buffer = save_audio_to_wav(np.frombuffer(audio_response_bytes, dtype=np.float32), tts_model.tts.synthesizer.output_sample_rate)
            return StreamingResponse(audio_buffer, media_type="audio/wav")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process audio segment: {str(e)}")
    finally:
        if temp_file_path:
            os.remove(temp_file_path)
