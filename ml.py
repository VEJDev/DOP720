import tempfile
import numpy as np
import pandas as pd
import io
import os
import pickle
import base64
from models import User, Procurement, UserProcurement
import keras
from keras import layers, models, optimizers, losses
import tensorflow as tf


class MachineLearning():
    RETRAIN_THRESHOLD = 200

    def __init__(self, database, application):
        self.db = database
        self.app = application

    def train(self, user_id, liked_ids): # liked_ids kā pass in mainīgais, lai varēt iztestēt ar test_ml.py, pēc vajadzēs noņemt
        with self.app.app_context():
            # pēc tam kad būs nepieciešams ņemt no datubāzes: 
            # liked_ids = self.db.session.query(
            #     UserProcurement.procurement_id).filter_by(user_id=user_id).all()
            # liked_ids = [id for (id,) in liked_ids]
            positive = self.db.session.query(Procurement.text, Procurement.customer).filter(
                Procurement.id.in_(liked_ids)
            ).all()

            negative = self.db.session.query(Procurement.text, Procurement.customer).filter(
                ~Procurement.id.in_(liked_ids)
            ).limit(len(positive)).all()

            X = [f"{text} {customer}" for text,
                 customer in positive + negative]
            y = [1] * len(positive) + [0] * len(negative)

            encoder = layers.TextVectorization(max_tokens=2000)
            encoder.adapt(tf.convert_to_tensor(X, dtype=tf.string))

            X_tensor = tf.convert_to_tensor(X, dtype=tf.string)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)

            model_data = self.getModel(user_id)

            if model_data is None or len(X) < self.RETRAIN_THRESHOLD:
                model = models.Sequential([
                    layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                                     output_dim=32,
                                     mask_zero=True),
                    layers.LSTM(32),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.4),
                    layers.Dense(1, activation='sigmoid')
                ])

                model.compile(optimizer=optimizers.Adam(learning_rate=0.005),
                              loss=losses.BinaryCrossentropy(),
                              metrics=['accuracy'])

                X_encoded = encoder(X_tensor)
                model.fit(X_encoded, y_tensor, epochs=5)
            else:
                model, encoder = self.deserialize_model_data(model_data)

                X_encoded = encoder(X_tensor)
                model.fit(X_encoded, y_tensor, epochs=5)

            self.saveModel(user_id, model, encoder)

    def predict(self, user_id, procurement_text):
        model_data = self.getModel(user_id)
        if model_data is None:
            raise ValueError(f"No model found for user {user_id}")

        model, encoder = self.deserialize_model_data(model_data)

        input_text = tf.convert_to_tensor([procurement_text], dtype=tf.string)

        encoded_input = encoder(input_text)
        return model.predict(encoded_input)

    def saveModel(self, user_id, model, encoder):
        with self.app.app_context():
            existing = self.db.session.query(
                User).filter_by(id=user_id).first()
            if existing:
                model_data = self.serialize_model_data(model, encoder)
                self.db.session.query(User).filter(User.id == user_id).update({
                    User.model: model_data
                })
                self.db.session.commit()

    def getModel(self, user_id):
        with self.app.app_context():
            user = self.db.session.query(User).filter(
                User.id == user_id).first()
            if user:
                return user.model
            return None

    def serialize_model_data(self, model, encoder):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model_weights.weights.h5")
            model.save_weights(model_path)

            with open(model_path, "rb") as f:
                model_bytes = f.read()

            encoder_config = encoder.get_config()
            vocab = encoder.get_vocabulary()

            data = {
                "model_bytes": model_bytes,
                "model_config": model.get_config(),
                "encoder_config": encoder_config,
                "encoder_vocab": vocab
            }

            serialized = pickle.dumps(data)
            return serialized

    def deserialize_model_data(self, model_data):
        data = pickle.loads(model_data)

        model_bytes = data["model_bytes"]
        model_config = data["model_config"]
        encoder_config = data["encoder_config"]
        vocab = data["encoder_vocab"]

        encoder = layers.TextVectorization.from_config(encoder_config)
        encoder.set_vocabulary(vocab)

        model = keras.Sequential.from_config(model_config)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.005),
                      loss=losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

        with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name

        try:
            model.load_weights(tmp_path)
            return model, encoder
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
