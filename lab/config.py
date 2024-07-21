import torch
print(torch.cuda.is_available())  # Deve retornar True se CUDA estiver disponível
print(torch.cuda.current_device())  # Deve retornar o índice da GPU ativa
print(torch.cuda.get_device_name(0))  # Deve retornar o nome da GPU
