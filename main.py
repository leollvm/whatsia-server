from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.llm_route import router as llm_router
from routes.text_chatbot_route import router as text_chatbot_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(llm_router)
app.include_router(text_chatbot_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
