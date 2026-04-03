from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
import joblib
import traceback
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import google.generativeai as genai 

# ==========================================
# 1. GEMINI API SETUP
# ==========================================
#  Note: Use Environment Variables for API Keys in production
genai.configure(api_key="AIzaSyAy0CGzXf3UZ14FmmFTiLC29tKpzr0BD_U")
model = genai.GenerativeModel('gemini-flash-latest')

app = Flask(__name__)
app.secret_key = "super_secret_key_for_session_security"

# ==========================================
# 2. DATABASE SETUP
# ==========================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/obesity_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ------------------------------------------
# 🗄️ MODELS
# ------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(150), nullable=False)
    assessments = db.relationship('Assessment', backref='user', lazy=True)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    age = db.Column(db.Float)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    prediction = db.Column(db.String(100))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==========================================
# 3. ML MODEL SETUP
# ==========================================
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop
    def fit(self, X, y=None): return self
    def transform(self, X):
        if self.features_to_drop:
            try:
                if isinstance(X, np.ndarray): 
                    try: X_df = pd.DataFrame(X, columns=feature_names)
                    except: X_df = pd.DataFrame(X)
                else: X_df = X
                return X_df.drop(columns=self.features_to_drop, errors="ignore").values
            except: return X
        return X

print("Loading ML models...")
pipeline = None
label_encoder = None
feature_names = []

try:
    if os.path.exists("models/obesity_best_pipeline.pkl"):
        pipeline = joblib.load("models/obesity_best_pipeline.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out"):
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        else:
            feature_names = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "Gender", "family", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
        print("✓ ML Models loaded successfully!")
    else:
        print("❌ Model files not found.")
except Exception as e:
    print(f"❌ Error loading ML models: {e}")

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def build_input_dataframe(form):
    try:
        age = float(form.get("Age"))
        height = float(form.get("Height"))
        weight = float(form.get("Weight"))
        fcvc = float(form.get("FCVC"))
        ncp = float(form.get("NCP"))
        ch2o = float(form.get("CH2O"))
        faf = float(form.get("FAF"))
        tue = float(form.get("TUE"))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Numeric field missing: {e}")

    row = {
        "Age": age, "Height": height, "Weight": weight, "FCVC": fcvc, "NCP": ncp,
        "CH2O": ch2o, "FAF": faf, "TUE": tue, "Gender": form.get("Gender"),
        "family_history_with_overweight": form.get("family_history_with_overweight"),
        "FAVC": form.get("FAVC"), "CAEC": form.get("CAEC"),
        "SMOKE": form.get("SMOKE"), "SCC": form.get("SCC"),
        "CALC": form.get("CALC"), "MTRANS": form.get("MTRANS"),
    }
    df = pd.DataFrame([row])
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["FCVC_FAF_Interact"] = df["FCVC"] * df["FAF"]
    df["CH2O_per_NCP"] = df["CH2O"] / (df["NCP"] + 1e-6)
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 18, 30, 45, 60, 120], labels=["Child", "Young Adult", "Adult", "Middle Age", "Senior"])
    return df, row

def explain_prediction(df_row, prediction_label):
    explanation = []
    bmi = float(df_row["BMI"].iloc[0])
    faf = float(df_row["FAF"].iloc[0])   
    fcvc = float(df_row["FCVC"].iloc[0]) 
    family = df_row["family_history_with_overweight"].iloc[0]
    
    if bmi >= 30: explanation.append(f"🔴 **High BMI:** Your BMI is {bmi:.1f} (Obese range).")
    elif bmi >= 25: explanation.append(f"🟡 **Overweight:** Your BMI is {bmi:.1f} (Above normal).")
    if faf < 1.0: explanation.append("📉 **Low Activity:** Lack of physical exercise detected.")
    if fcvc < 2.0: explanation.append("🥗 **Dietary Gap:** Low vegetable intake.")
    if family == "yes": explanation.append("🧬 **Genetics:** Family history increases risk.")
    if not explanation: explanation.append("ℹ️ Result based on overall lifestyle analysis.")
    return explanation

# ==========================================
# 5. ROUTES
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered!')
        else:
            new_user = User(name=name, phone=phone, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please login.')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    my_history = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.date.desc()).all()
    return render_template('history.html', assessments=my_history)

#  ADMIN ROUTE: View Users (Hides Admin from list)
@app.route('/admin')
@login_required
def admin():
    admin_email = "admin@gmail.com"
    if current_user.email == admin_email:
        # Filter out the admin account from the list
        users = User.query.filter(User.email != admin_email).all()
        return render_template('admin.html', users=users)
    else:
        return """
        <div style='text-align:center; margin-top:50px;'>
            <h1 style='color:red;'>⛔ Access Denied!</h1>
            <p>You do not have permission to view this page.</p>
            <a href='/'>Go Back Home</a>
        </div>
        """

#  DELETE ROUTE: Admin Only
@app.route('/delete_user/<int:user_id>')
@login_required
def delete_user(user_id):
    if current_user.email != 'admin@gmail.com':
        return "Access Denied"
    
    user_to_delete = User.query.get(user_id)
    if user_to_delete:
        # First delete all their assessments to avoid DB errors
        Assessment.query.filter_by(user_id=user_id).delete()
        
        # Then delete the user
        db.session.delete(user_to_delete)
        db.session.commit()
        flash('User deleted successfully!')
    
    return redirect(url_for('admin'))

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "GET":
        return render_template("index.html", user=current_user)

    try:
        if not pipeline: return "<h2>Error: Model not loaded.</h2>"
        
        df_input, raw_input = build_input_dataframe(request.form)
        encoded_pred = pipeline.predict(df_input)[0]
        prediction = label_encoder.inverse_transform([encoded_pred])[0]
        
        preprocessed = pipeline.named_steps["preprocessor"].transform(df_input)
        dropped = pipeline.named_steps["corr_drop"].transform(preprocessed)
        probs = pipeline.named_steps["model"].predict_proba(dropped)[0]
        class_probs = list(zip(label_encoder.classes_, probs))
        reasons = explain_prediction(df_input, prediction)

        if current_user.is_authenticated:
            try:
                new_test = Assessment(
                    user_id=current_user.id,
                    age=float(df_input["Age"].iloc[0]),
                    height=float(df_input["Height"].iloc[0]),
                    weight=float(df_input["Weight"].iloc[0]),
                    bmi=float(df_input["BMI"].iloc[0]),
                    prediction=prediction
                )
                db.session.add(new_test)
                db.session.commit()
                print("✅ Data Saved to History!")
            except Exception as e:
                print(f"❌ Error Saving to DB: {e}")

        session['prediction'] = str(prediction)
        session['bmi'] = float(df_input["BMI"].iloc[0])
        
        return render_template("result.html", 
                             prediction=prediction, 
                             class_probs=class_probs, 
                             user_input=raw_input, 
                             reasons=reasons, 
                             user=current_user)
    except Exception:
        traceback.print_exc()
        return f"<h2>Processing Error.</h2><pre>{traceback.format_exc()}</pre>"

@app.route("/chat_api", methods=["POST"])
@login_required
def chat_api():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message: return jsonify({"response": "Please type a message."})

        pred = session.get('prediction', 'Unknown')
        bmi = session.get('bmi', 'Unknown')
        user_name = current_user.name

        #  Knowledge 
        context = f"""
        Act as a professional health consultant. Base your advice strictly on these 6 books:
        
        1. 'The Obesity Code' by Dr. Jason Fung (Focus: Insulin resistance & Intermittent Fasting).
        2. 'Atomic Habits' by James Clear (Focus: Psychology of habit building).
        3. 'How Not to Die' by Dr. Michael Greger (Focus: Whole-food plant-based nutrition).
        4. 'Why We Sleep' by Matthew Walker (Focus: Impact of sleep on weight loss & metabolism).
        5. 'Metabolical' by Dr. Robert Lustig (Focus: Dangers of processed food & sugar).
        6. 'The Circadian Code' by Dr. Satchin Panda (Focus: Timing of eating & body clock).

        User Profile:
        - Name: {user_name}
        - Current Status: {pred}
        - BMI: {bmi}
        
        User Question: "{user_message}"
        
        Constraint: Keep answer concise (max 3-4 sentences) and use a motivating tone.
        """
        response = model.generate_content(context)
        reply = response.text.replace("*", "").strip()
        return jsonify({"response": reply})
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({"response": "I'm currently offline."})

if __name__ == "__main__":
    app.run(debug=True)