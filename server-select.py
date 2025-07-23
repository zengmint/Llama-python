from llama_cpp import Llama
from flask import Flask, request, jsonify
import os
import sys
# Diccionario de modelos disponibles
MODELS = {
    "phi":      r"D:\Programs\models\microsoft\Phi-3-mini-4k-instruct\Phi-3-Mini-4K-Instruct_Q8_0.gguf",
    "gemma":    r"D:\Programs\models\google\codegemma-7b-it\codegemma-7b-it_Q5_K_M.gguf",
    "hermes":   r"D:\Programs\models\NousResearch\Hermes-2-Pro-Mistral-7B\Hermes-2-Pro-Mistral-7B_Q6_K.gguf",
    "falcon":   r"D:\Programs\models\tiiuae\falcon-7b-instruct\Falcon-7B-Instruct_Q6_K.gguf"
}

def seleccionar_modelo_interactivo() -> str:
    """
    Muestra un menú interactivo para elegir modelo y devuelve la ruta del modelo elegido.
    """
    print("🎛️ Modelos disponibles:")
    opciones = list(MODELS.items())
    for i, (nombre, ruta) in enumerate(opciones, start=1):
        print(f"  {i}. {nombre.capitalize()} ({ruta})")

    while True:
        try:
            seleccion = int(input("\n🔍 Ingresá el número del modelo que querés cargar: "))
            if 1 <= seleccion <= len(opciones):
                nombre_modelo, ruta = opciones[seleccion - 1]
                if not os.path.exists(ruta):
                    print(f"❌ El archivo no se encontró en {ruta}")
                    continue
                print(f"\n✅ Modelo '{nombre_modelo}' seleccionado.\n")
                return ruta
            else:
                print("⚠️ Número fuera de rango. Intentá otra vez.")
        except ValueError:
            print("⚠️ Ingresá un número válido.")


def cargar_modelo_llama(model_path: str) -> Llama:
    """
    Carga un modelo GGUF con LlamaCpp desde la ruta dada.
    """
    print("🔄 Cargando modelo...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,  # Cambiar a -1 para usar GPU
        verbose=True
    )
    print("✅ Modelo cargado exitosamente.")
    return llm

# 🚀 Crear app Flask
app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    
    # Convertir mensajes a prompt simple
    prompt = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'user':
            prompt += f"User: {content}\n"
        elif role == 'assistant':
            prompt += f"Assistant: {content}\n"
    
    prompt += "Assistant: "
    
    # Generar respuesta
    response = llm(
        prompt,
        max_tokens=data.get('max_tokens', 512),
        temperature=data.get('temperature', 0.7),
        stop=["User:", "\n\n"]
    )
    
    return jsonify({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response['choices'][0]['text']
            }
        }]
    })

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get('prompt', '')
    
    response = llm(
        prompt,
        max_tokens=data.get('max_tokens', 512),
        temperature=data.get('temperature', 0.7)
    )
    
    return jsonify(response)

@app.route('/')
def home():
    return """
    <h1>🚀 Servidor LLaMA funcionando!</h1>
    <p>Endpoints disponibles:</p>
    <ul>
        <li><b>/v1/chat/completions</b> - Para chat</li>
        <li><b>/v1/completions</b> - Para completar texto</li>
    </ul>
    """

if __name__ == "__main__":
    # Flask recarga automáticamente el script si debug=True, así que evitamos
    # que vuelva a pedir el modelo con este chequeo:

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        MODEL_PATH = seleccionar_modelo_interactivo()
        llm = cargar_modelo_llama(MODEL_PATH)
    
    print("🚀 Servidor iniciado en http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
