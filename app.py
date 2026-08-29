import pymysql
pymysql.install_as_MySQLdb()
import os
import markdown
import re
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-3.6-flash')
from flask import Flask, render_template, request, redirect, session
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'greenguard_secret_key')

# MySQL configuration
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'greenguard_db')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
mysql = MySQL(app)

# Load CNN model and labels once at startup
model = tf.keras.models.load_model('model/greenguard_model.h5')
with open('model/class_labels.json', 'r') as f:
    class_labels = json.load(f)

@app.route('/')
def home():
    return "GreenGuard is running!"

# ---------- SIGNUP ----------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_password))
        mysql.connection.commit()
        cur.close()

        return redirect('/login')

    return render_template('signup.html')

# ---------- LOGIN ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            return redirect('/dashboard')
        else:
            return "Invalid email or password."

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT predicted_class, confidence, prediction_date 
        FROM predictions 
        WHERE user_id = %s AND predicted_class NOT LIKE '%%healthy%%'
        ORDER BY prediction_date DESC LIMIT 3
    """, (session['user_id'],))
    alerts = cur.fetchall()
    cur.close()

    return render_template('dashboard.html', user_name=session['user_name'], alerts=alerts)

# ---------- UPLOAD PAGE ----------
@app.route('/upload')
def upload():
    return render_template('upload.html')

# ---------- PREDICT ----------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['leaf_image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_labels[str(predicted_index)]
    confidence = round(float(np.max(predictions[0])) * 100, 2)

    def get_disease_info(disease_name):
        try:
            response = gemini_model.generate_content(
                f"In simple bullet points, give: Causes, Symptoms, Prevention, and Treatment for the plant disease '{disease_name}'. Keep it concise, under 150 words total."
            )
            raw_text = re.sub(r'\n(?=\*)', '\n\n', response.text)
            return markdown.markdown(raw_text)
        except Exception as e:
            print("GEMINI ERROR:", e)
            return "Disease information currently unavailable."

    disease_info = get_disease_info(predicted_class)

    # Save to prediction history (if logged in)
    if 'user_id' in session:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO predictions (user_id, image_path, predicted_class, confidence) VALUES (%s, %s, %s, %s)",
                    (session['user_id'], filepath, predicted_class, confidence))
        mysql.connection.commit()
        cur.close()

    return render_template('result.html', prediction=predicted_class, confidence=confidence, 
                        image_path='/' + filepath, disease_info=disease_info)
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user_name=session['user_name'])


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect('/login')
    cur = mysql.connection.cursor()
    cur.execute("SELECT image_path, predicted_class, confidence, prediction_date FROM predictions WHERE user_id = %s ORDER BY prediction_date DESC", (session['user_id'],))
    records = cur.fetchall()
    cur.close()
    return render_template('history.html', records=records)

@app.route('/admin')
def admin_dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    cur = mysql.connection.cursor()
    cur.execute("SELECT role FROM users WHERE id = %s", (session['user_id'],))
    role = cur.fetchone()[0]

    if role != 'admin':
        return "Access Denied: Admins only."

    cur.execute("SELECT id, name, email, password, role FROM users")
    users = cur.fetchall()

    cur.execute("SELECT user_id, predicted_class, confidence, prediction_date FROM predictions ORDER BY prediction_date DESC")
    predictions = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cur.fetchone()[0]

    cur.close()

    return render_template('admin_dashboard.html', users=users, predictions=predictions,
                            total_users=total_users, total_predictions=total_predictions)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)