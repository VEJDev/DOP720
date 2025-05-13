from flask import Flask, request, jsonify
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
    records = db.session.query(Procurement).all()
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    return render_template('login.html')

if __name__ == '__main__':
    init_db()
    app.run()
    #scraper.run()