import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Paths
MODEL_PATH = 'retention_action_model.pkl'
DATA_PATH = 'Hr_Retention.csv'

# Global variables for model & metadata
model_obj = None
tfidf_vectorizer = None
trained_svm = None
class_labels = ['Salary Hike', 'Flexible Work', 'Manager Change', 'Promotion', 'Role Change', 'Not Applicable']
action_strategies = {
    'Salary Hike': {
        'title': 'Compensation & Financial Incentive Adjustment',
        'badge_class': 'badge-salary',
        'strategy': 'Conduct an immediate market compensation review. Prepare a competitive salary adjustment or retention bonus proposal to align with industry benchmarks.',
        'urgency': 'High',
        'icon': 'fa-sack-dollar'
    },
    'Flexible Work': {
        'title': 'Work-Life Balance & Remote Work Arrangement',
        'badge_class': 'badge-flexible',
        'strategy': 'Offer hybrid or fully remote work options, flexible working hours, or reduced shift hours to alleviate burnout and improve work-life harmony.',
        'urgency': 'Medium',
        'icon': 'fa-house-laptop'
    },
    'Manager Change': {
        'title': 'Team Transfer & Leadership Alignment',
        'badge_class': 'badge-manager',
        'strategy': 'Initiate an internal transfer to a new reporting manager or department. Conduct confidential 1-on-1 feedback sessions to resolve team dynamic issues.',
        'urgency': 'High',
        'icon': 'fa-users-gear'
    },
    'Promotion': {
        'title': 'Career Progression & Title Upgrade',
        'badge_class': 'badge-promotion',
        'strategy': 'Accelerate the performance review cycle. Provide a clear career advancement roadmap, title promotion, and leadership opportunity within the project.',
        'urgency': 'High',
        'icon': 'fa-arrow-trend-up'
    },
    'Role Change': {
        'title': 'Internal Job Transfer & Skill Realignment',
        'badge_class': 'badge-role',
        'strategy': 'Explore internal job rotation to align with the employee’s skill set and long-term career aspirations. Offer specialized domain training.',
        'urgency': 'Medium',
        'icon': 'fa-repeat'
    },
    'Not Applicable': {
        'title': 'Exit Review & Knowledge Transfer',
        'badge_class': 'badge-na',
        'strategy': 'The attrition reason relates to non-negotiable personal factors (e.g. relocation, family commitments). Focus on smooth offboarding and knowledge transfer.',
        'urgency': 'Low',
        'icon': 'fa-circle-info'
    }
}

def load_and_initialize_model():
    global model_obj, tfidf_vectorizer, trained_svm, class_labels

    print(f"Loading {MODEL_PATH}...")
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_obj = pickle.load(f)
        print(f"Loaded model object type: {type(model_obj)}")

    # Load dataset to fit vectorizer and train model if model_obj is predictions array
    if os.path.exists(DATA_PATH):
        print(f"Loading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        data = df[df['Attrition'] == 'Yes'].dropna(subset=['Exit_Reason_HR_Recorded', 'Retention_Action_Taken'])
        
        X = data['Exit_Reason_HR_Recorded']
        y = data['Retention_Action_Taken']
        
        class_labels = sorted(list(y.unique()))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        
        trained_svm = SVC(kernel='linear', decision_function_shape='ovr', class_weight='balanced', probability=True, random_state=42)
        trained_svm.fit(X_train_tfidf, y_train)
        print("Model vectorizer and classifier ready!")

load_and_initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json(silent=True) or request.form
        text = req_data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Please provide text/feedback to predict.'}), 400

        # Predict using pipeline or vectorizer + SVM
        if hasattr(model_obj, 'predict'):
            prediction = model_obj.predict([text])[0]
            if hasattr(model_obj, 'predict_proba'):
                probs = model_obj.predict_proba([text])[0]
                prob_dict = {cls: float(p) for cls, p in zip(model_obj.classes_, probs)}
            else:
                prob_dict = {cls: (1.0 if cls == prediction else 0.0) for cls in class_labels}
        elif tfidf_vectorizer is not None and trained_svm is not None:
            text_vectorized = tfidf_vectorizer.transform([text])
            prediction = trained_svm.predict(text_vectorized)[0]
            probs = trained_svm.predict_proba(text_vectorized)[0]
            prob_dict = {cls: float(p) for cls, p in zip(trained_svm.classes_, probs)}
        else:
            # Fallback
            prediction = 'Not Applicable'
            prob_dict = {cls: 0.16 for cls in class_labels}

        # Calculate confidence score percentage
        max_prob = prob_dict.get(prediction, 0.0)
        confidence_pct = round(max_prob * 100, 1)

        # Get strategy details
        strategy_info = action_strategies.get(prediction, action_strategies['Not Applicable'])

        # Prepare probability breakdown sorted
        prob_breakdown = [
            {
                'action': cls,
                'probability': round(prob_dict.get(cls, 0.0) * 100, 1),
                'badge_class': action_strategies.get(cls, {}).get('badge_class', 'badge-na')
            }
            for cls in sorted(prob_dict.keys(), key=lambda x: prob_dict[x], reverse=True)
        ]

        return jsonify({
            'status': 'success',
            'input_text': text,
            'prediction': prediction,
            'confidence': confidence_pct,
            'strategy_info': strategy_info,
            'probabilities': prob_breakdown
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': f"Error processing prediction: {str(e)}"}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            total_records = len(df)
            attrition_count = len(df[df['Attrition'] == 'Yes'])
            retained_count = len(df[df['Retained'] == 'Yes']) if 'Retained' in df.columns else 0
            
            action_counts = df[df['Attrition'] == 'Yes']['Retention_Action_Taken'].value_counts().to_dict()
            
            return jsonify({
                'total_records': total_records,
                'attrition_count': attrition_count,
                'retained_count': retained_count,
                'action_distribution': action_counts
            })
        return jsonify({'message': 'Dataset stats unavailable'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/samples', methods=['GET'])
def get_samples():
    samples = [
        {
            'category': 'Compensation & Pay',
            'text': 'Employee felt underpaid compared to market standards and requested a salary review after taking on additional responsibilities.'
        },
        {
            'category': 'Work-Life Balance / Burnout',
            'text': 'Reported severe burnout due to high workload, mandatory weekend shifts, and continuous long working hours over the past 6 months.'
        },
        {
            'category': 'Managerial Conflict',
            'text': 'Cited lack of support, poor communication, and recognition issues with the direct team manager.'
        },
        {
            'category': 'Career Advancement',
            'text': 'Felt there were no growth opportunities or title promotions available within the current project team.'
        },
        {
            'category': 'Relocation / Personal',
            'text': 'Spouse received a job transfer to another state requiring family relocation.'
        }
    ]
    return jsonify(samples)

if __name__ == '__main__':
    print("Starting HR Retention Action Prediction Web Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
