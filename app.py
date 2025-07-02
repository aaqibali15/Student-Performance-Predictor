from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import joblib

# Define interaction feature logic
def add_interaction_features(X):
    X = X.copy()
    X['Effective Study Time'] = X['Study Hours'] * X['Concentration Level'] / (X['Distraction Time'] + 1)
    X['Revision Boost'] = (X['Revision Done'] == 'Yes').astype(int) * 5
    X['Sleep Efficiency'] = X['Sleep Hours'] / (X['Distraction Time'] + 1)
    return X

# Load ML model
model = joblib.load('student_score_model.pkl')



app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Define the Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

with app.app_context():
    db.create_all()

# Set signup page as home page
@app.route('/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not (name and email and password and confirm_password):
            flash('Please fill out all fields.', 'warning')
            return render_template('signup.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'warning')
            return render_template('signup.html')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please use a different email.', 'warning')
            return render_template('signup.html')  # <-- Render signup again, not redirect

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not (email and password):
            flash('Please enter both email and password.', 'login_error')
            return render_template('login.html')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'login_error')
            return render_template('login.html')

    return render_template('login.html')


# Dashboard route after login
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', username=user.name)


# Route to view old predictions
@app.route('/old_predictions')
def view_predictions():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('view_predictions.html', predictions=user.predictions)

from flask import abort, flash, redirect, url_for, session

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    prediction = Prediction.query.get_or_404(prediction_id)

    if prediction.user_id != session['user_id']:
        abort(403)  # Forbidden access

    try:
        db.session.delete(prediction)
        db.session.commit()
        # flash('Prediction deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting prediction: {e}', 'danger')

    return redirect(url_for('view_predictions'))


# Route for new prediction input form

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user_id" not in session:
        flash("Please log in to make a prediction.", "warning")
        return redirect(url_for('login'))

    if request.method == "POST":
        try:
            input_data = {
                "Study Hours": float(request.form["study_hours"]),
                "Sleep Hours": float(request.form["sleep_hours"]),
                "Topics to Cover": int(request.form["topics"]),
                "Previous Test Score": float(request.form["previous_score"]),
                "Concentration Level": float(request.form["concentration"]),
                "Distraction Time": float(request.form["distraction"]),
                "Time Left for Exam": int(request.form["time_left"]),
                "Practice Tests Taken": int(request.form["practice_tests"]),
                "Assignment Submitted": request.form["assignment"],
                "Topic Difficulty": request.form["difficulty"],
                "Learning Capacity": request.form["capacity"],
                "Class Participation": request.form["participation"],
                "Health Status": request.form["health"],
                "Study Environment": request.form["environment"],
                "Revision Done": request.form["revision"],
                "Topic Familiarity": request.form["familiarity"]
            }

            df = pd.DataFrame([input_data])

            # Add interaction features if needed
            # df = add_interaction_features(df)

            # Get raw prediction and clamp between 0-100
            raw_prediction = model.predict(df)[0]
            clamped_prediction = max(0.0, min(float(raw_prediction), 100.0))
            score = round(clamped_prediction, 2)

            # Message logic remains the same
            if score >= 90:
                message = "🌟 Excellent work! Your dedication is truly commendable! 🚀"
                tip = "Tip: Keep maintaining your study schedule and revise regularly to stay ahead!"
            elif score >= 80:
                message = "🔥 Great job! You're on the path to success! Keep pushing!"
                tip = "Tip: Focus on reviewing weaker topics to achieve an even higher score!"
            elif score >= 70:
                message = "✅ Good effort! Some improvement needed but you’re doing well!"
                tip = "Tip: Try increasing your practice test attempts for better retention."
            elif score >= 60:
                message = "⚡ Keep going! You have potential to shine even brighter!"
                tip = "Tip: Improve your concentration by minimizing distractions during study sessions."
            else:
                message = "💡 Don't be discouraged! Keep working hard and you'll improve!"
                tip = "Tip: Work on fundamental concepts and practice frequently. Small improvements lead to big success!"

            # Save clamped prediction
            prediction_record = Prediction(
                user_id=session['user_id'],
                input_data=str(input_data),
                result=str(score)  # Now stores the clamped value
            )
            db.session.add(prediction_record)
            db.session.commit()

            return render_template("result.html", prediction=score, message=message, tip=tip)

        except Exception as e:
            flash(f"Error processing input: {e}", "danger")
            return redirect(url_for('predict'))

    return render_template("index.html")


# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    # flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
