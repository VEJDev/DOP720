import numpy as np
import pandas as pd
import io
import os
from models import User, Procurement

# Versija atkarīga no python versijas
import tensorflow as tf # pip install tensorflow

# Nav testēts
class MachineLearning():
    def __init__(self, database, application):
        global db, app, encoder
        db = database
        app = application

        with app.app_context():
            if not os.path.exists("encoder_model"):
                procurements = db.session.query(Procurement.text, Procurement.customer).all()
                all_records = [f"{text} {customer}" for text, customer in procurements]
                encoder = tf.keras.layers.TextVectorization(max_tokens=2000)
                encoder.adapt(all_records)
                tf.keras.models.save_model(encoder, 'encoder_model')
            encoder = tf.keras.models.load_model('encoder_model')

    def train(user_id, train_data):
        if getModel(user_id) is not None:
            model = tf.keras.Sequential([
                encoder,
                tf.keras.layers.Embedding(
                    input_dim=len(encoder.get_vocabulary()),
                    output_dim=32,
                    mask_zero=True
                ),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
            saveModel(user_id, model)
        model = getModel(user_id)
        model.fit(train_data, epochs=5)
        saveModel(user_id, model)

    def predict(user_id, procurement_text):
        model = getModel(user_id)
        model.predict(procurement_text)

def saveModel(user_id, model):
    with app.app_context():
        existing = db.session.query(User).filter_by(id=user_id).first()
        if existing:
            db.session.query(User).filter(id == user_id).update({
                User.model: serialize_model(model)
            })
        db.session.commit()

def getModel(user_id):
    with app.app_context():
        return db.session.query(User).filter(User.id == user_id).first().model

def serialize_model(model):
    byte_io = io.BytesIO()
    tf.keras.models.save_model(model, byte_io, save_format='h5')
    return byte_io.getvalue()

def deserialize_model(model_bytes):
    byte_io = io.BytesIO(model_bytes)
    return tf.keras.models.load_model(byte_io)