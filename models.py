from flask import Flask, request, redirect, url_for, render_template, flash, get_flashed_messages

import os
from models import Procurement
from models import init_db
from flask_sqlalchemy import SQLAlchemy # pip install Flask-SQLAlchemy
from flask import render_template
from scraper import ProcurementScraper
from ml import MachineLearning

from werkzeug.security import generate_password_hash
from sqlalchemy.exc import IntegrityError
from models import User

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///{}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db"))
db = SQLAlchemy(app)
scraper = ProcurementScraper(db, app)
ml = MachineLearning(db, app)
app.secret_key = 'f3ac29d2b0c145b7ad03cfbde72a7810'

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email,
                        password=hashed_password, model=None)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect('/login')
        except IntegrityError:
            db.session.rollback()
            flash('Username or email already exists!', 'error')

    return render_template('register.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/suggestions')
def suggestions():
    return render_template('suggestions.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
    # scraper.run()
