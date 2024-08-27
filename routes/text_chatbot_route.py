import json
import redis
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from models.llm_model import LLMModel

router = APIRouter()

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
llm_model = LLMModel()

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

@router.post("/chatbot")
async def chatbot(session_id: str = Form(...), query: str = Form(...), image_code: str = Form(...), language_code: str = Form(...)):
    try:
        print(f"Received query: {query}, session_id: {session_id}, image_code: {image_code}")
        print(language_code)

        if not session_id or not query or not image_code:
            raise HTTPException(status_code=400, detail="session_id, query, and image_code are required")

        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'pt': 'Portuguese'
        }
        language = language_map.get(language_code, 'English')

        guardian_map = {
            '1': ('Sakura', 'Feminine'),
            '2': ('Mio', 'Male')
        }
        nameGuardian, genderGuardian = guardian_map.get(image_code, ('Sakura', 'Feminine'))

        conversation_context = get_session(session_id)
        if not conversation_context:
            conversation_context = [
                " These are your premises: follow them, only respond to what is said after your premises"
                + " limit your answer to 20 words maximum,"
                + " you are an " + language + " language teacher, your mission is to help me learn more, correct my mistakes in a friendly way,"
                + " your name is " + nameGuardian + ', ' + genderGuardian
                + " also correct me if I conjugate the wrong verb, help me to be fluent in " + language
                + " Don't repeat my speech, just go straight to the answer,"
                + " help me by encouraging me to talk to you more,"
                + " If my sentence is strange, try to understand the context of our conversation and ask if something similar to the words I said is actually correct,"
                + " If you create a list, do not number it or use symbols, instead, add commas and skip 2 lines"
                + " put a small pause when there are line breaks: "
                + " you should prefer to respond in " + language + " however, if you see that I am having difficulty, respond in the language I am speaking to you. (end of premises) "
            ]

        conversation_context.append(query)
        set_session(session_id, conversation_context)

        full_context = " ".join([str(item) for item in conversation_context if isinstance(item, str)])
        response = llm_model.invoke(full_context)

        print("Ollama response:", response)

        conversation_context.append(response)
        set_session(session_id, conversation_context)

        return JSONResponse(content={"response": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
