import tempfile
import numpy as np
import os
import pickle

from models import User, Procurement

import tensorflow as tf
# importē no tf-keras, ne no keras, ja gribi kopā ar BERT
# pip install huggingface_hub[hf_xet], priekš labākas veiktspējas
from tf_keras import layers, models, optimizers, losses
from transformers import BertTokenizer, TFBertModel


class MachineLearning:
    RETRAIN_THRESHOLD = 500

    def __init__(self, db, app):
        self.db = db
        self.app = app
        self.model_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = TFBertModel.from_pretrained(self.model_name)

    def get_bert_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True,
                                truncation=True, return_tensors="tf")
        outputs = self.bert_model(inputs)
        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
        return embeddings.numpy()

    def train(self, user_id, liked_ids):
        with self.app.app_context():
            positive = self.db.session.query(Procurement.text).filter(
                Procurement.id.in_(liked_ids)
            ).all()

            negative = self.db.session.query(Procurement.text).filter(
                ~Procurement.id.in_(liked_ids)
            ).limit(len(positive)).all()

            X = [text for (text,) in positive + negative]
            y = np.array([1] * len(positive) + [0] * len(negative))

            X_emb = self.get_bert_embeddings(X)
            model = models.Sequential([
                layers.Input(shape=(X_emb.shape[1],)),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam',
                          loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_emb, y, epochs=5)

            model.save(f"user_model_{user_id}.h5")

    def predict(self, user_id, procurement_text):
        model_path = f"user_model_{user_id}.h5"
        if not tf.io.gfile.exists(model_path):
            raise ValueError(f"Modelis lietotājam {user_id} nav saglabāts")

        model = models.load_model(model_path)

        emb = self.get_bert_embeddings([procurement_text])

        pred = model.predict(emb)
        return pred[0][0]
