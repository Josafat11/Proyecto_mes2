import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import os  # 👈 Importar os para leer la variable de entorno PORT

# Cargar modelo y scaler
modelo = joblib.load('modelo_rfr.pkl')
scaler = joblib.load('scaler_modelo.pkl')

# Lista de atributos 
atributos = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'Age']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Extraer datos del formulario
        entrada = [float(request.form[attr]) for attr in atributos]

        # Escalar entrada
        entrada_scaled = scaler.transform([entrada])

        # Predicción
        resultado = modelo.predict(entrada_scaled)[0]

        return render_template('formulario.html', prediccion=f'Fuerza estimada: {resultado:.2f} MPa')
    except:
        return render_template('formulario.html', prediccion='Error en los datos de entrada')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
