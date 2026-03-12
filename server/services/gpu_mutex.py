import threading

# Un candado global maestro para evitar que múltiples hilos disparen kernels
# de PyTorch de manera asíncrona hacia la GPU (GTX 1050 Ti) al mismo tiempo.
# Esto previene permanentemente el "CUDA error: illegal memory access"
# causado por la colisión asíncrona de los descriptores de VRAM.
global_gpu_lock = threading.Lock()
