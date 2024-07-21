from langchain_community.llms import Ollama

class LLMModel:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434", device="cuda"):
        self.model = Ollama(model=model_name, base_url=base_url)
        self.device = device
        self._configure_device()

    def _configure_device(self):
        # Aqui você precisaria configurar a biblioteca subjacente para usar a GPU
        # Este é um exemplo hipotético, pois depende da implementação da biblioteca Ollama
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

    def invoke(self, context):
        return self.model.invoke(context)
