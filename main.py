from flask import Flask, request, redirect, url_for
import os
from models import Procurement
from models import init_db
from flask_sqlalchemy import SQLAlchemy
from flask import render_template

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///{}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db"))
db = SQLAlchemy(app)

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
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
    app.run()
    #scraper.run()