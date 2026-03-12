"""
seed_all.py — Ejecuta seed_students y seed_uniform en orden.
"""
import os
import runpy
import sys

_server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_server_dir)
sys.path.insert(0, _server_dir)

import socket

def is_server_running(port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if __name__ == "__main__":
    if is_server_running():
        print("\n\033[91m[ERROR CRÍTICO EVITADO] ¡El servidor principal está corriendo en este contenedor!\033[0m")
        print("ChromaDB con SQLite (Persistente) NO permite que dos procesos distintos abran la base de datos al mismo tiempo, de lo contrario se corrompe (error 'table embeddings already exists').")
        print("\n\033[93mSOLUCIÓN PERMANENTE:\033[0m")
        print("Para ejecutar el seed de forma segura, primero debes apagar el contenedor y ejecutarlo en un entorno aislado con este comando en tu consola de Windows:")
        print("  docker-compose stop; docker-compose run --rm pyimage-service python scripts/seed_all.py; docker-compose start\n")
        sys.exit(1)

    print("=" * 65)
    print("  SEED COMPLETO — Estudiantes + Uniformes")
    print("=" * 65)
    
    import subprocess
    
    subprocess.check_call([sys.executable, os.path.join(_server_dir, "scripts", "seed_students.py")])
    print()
    subprocess.check_call([sys.executable, os.path.join(_server_dir, "scripts", "seed_uniform.py")])
