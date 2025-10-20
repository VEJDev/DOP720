from flask import Flask, request, redirect, url_for, render_template, flash, get_flashed_messages, session

import os
from datetime import datetime
from models import Procurement, UserProcurement
from models import init_db
from flask_sqlalchemy import SQLAlchemy # pip install Flask-SQLAlchemy
from sqlalchemy.dialects.sqlite import insert
from flask import render_template
from scraper import ProcurementScraper
from ml import MachineLearning

from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from models import User

import tensorflow as tf
from tf_keras import models
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///{}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db"))
db = SQLAlchemy(app)
ml = MachineLearning(db, app)
app.secret_key = '' # Enter your own secret key
scraper = ProcurementScraper(db, app)

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('suggestions'))
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('suggestions'))
    
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password']

        user = db.session.query(User).filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('suggestions'))
        else:
            flash('Invalid email or password!', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/deleteprofile')
def deleteprofile():
    if 'user_id' in session and 'username' in session:
        try:
            db.session.delete(db.session.query(User).filter_by(id=session.get('user_id', None)).first())
            db.session.commit() 
            session.clear()
        except Exception as e: print(e)
    return redirect(url_for('login'))

@app.route('/retrain')
def retrain():
    if 'user_id' in session and 'username' in session:
        try:
            user_id = session.get('user_id')
            user = db.session.query(User).filter_by(id=user_id).first()
            user.model = None
            db.session.query(UserProcurement).filter_by(
                user_id=user_id).delete()
            db.session.commit()

            model_path = f"ml_models/user_model_{user_id}.h5"
            if os.path.exists(model_path):
                os.remove(model_path)           
            return redirect(url_for('profile'))

        except Exception as e:
            print(e)
    else:
        return redirect(url_for('login'))


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

@app.route('/training', methods=['GET', 'POST'])
@app.route('/training/<int:page>', methods=['GET', 'POST'])
def training(page=1):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        selected_records = request.form.getlist('selected_records')
        
        
        all_records = request.form.getlist('all_records')
        selected_records = [int(record_id) for record_id in selected_records]
        all_records = [int(record_id) for record_id in all_records]
        unselected_records = [record_id for record_id in all_records if record_id not in selected_records]
        
        logger.debug(f"{selected_records} seleecteddd")
        logger.debug(f"{unselected_records} unselected_records")

        
        user_procurements = [
            {'user_id': session.get('user_id', None), 'procurement_id': record}
            for record in all_records
        ]

        stmt = insert(UserProcurement).values(user_procurements)
        stmt = stmt.on_conflict_do_nothing(index_elements=['user_id', 'procurement_id']) 
        db.session.execute(stmt)
        db.session.commit()

        ml.train(session.get('user_id', None), selected_records, unselected_records)
        return redirect(url_for('training'))

    current_time = datetime.now()
    start_index = (page - 1) * 20
    end_index = start_index + 20

    procurements = db.session.query(Procurement) \
        .filter(Procurement.deadline < current_time) \
        .filter(~Procurement.id.in_(
            db.session.query(UserProcurement.procurement_id)
            .filter(UserProcurement.user_id == session['user_id'])
        )) \
        .limit(20) \
        .offset(start_index) \
        .all()
    
    total_count = db.session.query(Procurement) \
        .filter(Procurement.deadline < current_time) \
        .filter(~Procurement.id.in_(
            db.session.query(UserProcurement.procurement_id)
            .filter(UserProcurement.user_id == session['user_id'])
        )) \
        .count()

    total_pages = (total_count // 20) + (1 if total_count % 20 > 0 else 0)
    return render_template('training.html', procurements=procurements, page=page, total_pages=total_pages, total_records=total_count)

@app.route('/suggestions', methods=['GET'])
@app.route('/suggestions/<int:page>', methods=['GET'])
def suggestions(page=1):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    current_time = datetime.now()
    start_index = (page - 1) * 20
    end_index = start_index + 20 - 1

    procurements = db.session.query(Procurement).filter(Procurement.deadline > current_time) \
        .limit(20) \
        .offset(start_index) \
        .all()

    total_count = db.session.query(Procurement).filter(Procurement.deadline > current_time).count()
    total_pages = (total_count // 20) + (1 if total_count % 20 > 0 else 0)
    
    output_procurements = []
    user_id = session.get('user_id', None)

    error_text = "Neviens iepirkums netika atrasts."

    try:
        model_path = f"ml_models/user_model_{user_id}.h5"
        model = models.load_model(model_path)
        for procurement in procurements:
            output_procurements.append({
                'procurement': procurement,
                'score': round(ml.predict(model=model, procurement_text=procurement.text), 2)
            })

    except Exception as e:
        error_text = "Modelis nav apmācīts"
        total_count = 0
        print(f"An error occurred: {e}")

    return render_template('suggestions.html', procurements=output_procurements, page=page, total_pages=total_pages, start_index=start_index, end_index=end_index, total_records=total_count, error_text=error_text)
@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html', username=session.get('username', None))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

