from flask import Flask, request, render_template, redirect, url_for,jsonify
import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
import os
import webview
import threading
import serial
import csv
from datetime import datetime

# Serial config
COM_PORT = 'COM10'
BAUD_RATE = 115200
CSV_FILE = 'serial_data.csv'
serial_thread = None
serial_active = False

#webview.create_window("My App", "http://127.0.0.1:5000")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
def load_model():
    model_path = 'model/brain.h5'
    
    def expand_dims_layer(x):
        return tf.expand_dims(x, axis=2)
    
    def custom_output_shape(input_shape):
        return (input_shape[0], input_shape[1], 1)
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'Lambda': tf.keras.layers.Lambda(expand_dims_layer, output_shape=custom_output_shape)
            },
            compile=False
        )
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

def read_serial():
    global serial_active
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening serial port {COM_PORT}: {e}")
        return

    try:
        with open(CSV_FILE, 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Signal'])
    except FileExistsError:
        pass

    print("Serial reading started...")

    while serial_active:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            if line:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(CSV_FILE, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, line])
                print(f"Timestamp: {timestamp} -> Signal: {line}")

    ser.close()
    print("Serial reading stopped.")

@app.route('/start_serial')
def start_serial():
    global serial_thread, serial_active
    if not serial_active:
        serial_active = True
        serial_thread = threading.Thread(target=read_serial)
        serial_thread.start()
    return redirect(url_for('index'))


@app.route('/stop_serial')
def stop_serial():
    global serial_active
    serial_active = False
    return redirect(url_for('index'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = request.form.get('input_data')
        input_data = list(map(float, input_data.split(',')))
        input_data = np.array(input_data).reshape(1, 8)
        prediction = model.predict(input_data)
        return render_template('index.html', prediction=prediction[0][0])
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('display_text', filename=file.filename))
    return render_template('upload.html')

@app.route('/serial_data')
def serial_data():
    try:
        with open(CSV_FILE, 'r') as f:
            lines = f.readlines()[-10:]  # Get last 10 lines
        data = [line.strip().split(',') for line in lines[1:]]  # skip header
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/display/<filename>')
def display_text(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    extracted_text = df.to_html()
    return render_template('display.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)
   #webview.start()
