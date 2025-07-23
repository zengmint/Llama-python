from llama_cpp import Llama
from flask import Flask, request, jsonify
import os

# ⚠️ Ruta exacta a tu modelo GGUF
MODEL_PATH = "D:\Programs\models\microsoft\Phi-3-mini-4k-instruct/Phi-3-Mini-4K-Instruct_Q8_0.gguf"

# Verificar que el modelo existe
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: No se encontró el modelo en {MODEL_PATH}")
    exit(1)

# 🧠 Cargar modelo
print("🔄 Cargando modelo...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=0,  # Cambia a -1 para GPU
    verbose=True
)

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
    print("🚀 Servidor iniciado en http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)


    