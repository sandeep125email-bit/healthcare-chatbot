
from flask import Flask, render_template, request, session
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'healthcarechatbotsecret'

# Load updated model and dataset
model = joblib.load('C:\\Users\\HP\\Downloads\\Batch_5\\Batch_5\\Healthcare_Chatbot_Final_Project\\Healthcare_Chatbot_Final_Project\\healthcare_chatbot_model_sklearn_1_4.pkl')
df = pd.read_csv('C:\\Users\\HP\\Downloads\\Batch_5\\Batch_5\\Healthcare_Chatbot_Final_Project\Healthcare_Chatbot_Final_Project\\ai_healthcare_chatbot_full_dataset.csv')
@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['symptoms']
    prediction = model.predict([user_input])[0]

    match_row = df[df['symptoms'].str.lower().str.contains(user_input.lower())]
    if not match_row.empty:
        row = match_row.sample(1).iloc[0]
    else:
        row = df[df['predicted_disease'] == prediction].sample(1).iloc[0]

    chat_entry = {
        'user_input': user_input,
        'diagnosis': prediction,
        'medication': row['otc_medication'],
        'recommendation': row['recommendation']
    }
    history = session.get('chat_history', [])
    history.append(chat_entry)
    session['chat_history'] = history

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
