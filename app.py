from flask import Flask, render_template, request, session
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'healthcarechatbotsecret'

# Get the current file path (works on both Windows & Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and dataset using relative paths
model_path = os.path.join(BASE_DIR, 'healthcare_chatbot_model_sklearn_1_4.pkl')
data_path = os.path.join(BASE_DIR, 'ai_healthcare_chatbot_full_dataset.csv')

# Load model and data
model = joblib.load(model_path)
df = pd.read_csv(data_path)

@app.route('/')
def index():
    # Initialize chat history if not already in session
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    user_input = request.form['symptoms']

    # Make prediction using the model
    prediction = model.predict([user_input])[0]

    # Find the matching row from dataset
    match_row = df[df['symptoms'].str.lower().str.contains(user_input.lower())]
    if not match_row.empty:
        row = match_row.sample(1).iloc[0]
    else:
        row = df[df['predicted_disease'] == prediction].sample(1).iloc[0]

    # Store chat entry
    chat_entry = {
        'user_input': user_input,
        'diagnosis': prediction,
        'medication': row['otc_medication'],
        'recommendation': row['recommendation']
    }

    # Update chat history
    history = session.get('chat_history', [])
    history.append(chat_entry)
    session['chat_history'] = history

    # Re-render page with updated chat
    return render_template('index.html', chat_history=session['chat_history'])

if __name__ == '__main__':
    # Only runs locally; Render will use gunicorn from Procfile
    app.run(host='0.0.0.0', port=5000, debug=True)

