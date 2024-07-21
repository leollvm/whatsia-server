import whisper

class ASRModel:
    def __init__(self, model_name="small"):
        self.model = whisper.load_model(model_name).to("cuda")

    def transcribe(self, file_path):
        return self.model.transcribe(file_path)
