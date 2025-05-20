import tempfile
import numpy as np
import os
import pickle
import stanza

from models import User, Procurement

import tensorflow as tf
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

        stanza.download("lv") 
        self.nlp = stanza.Pipeline(
            lang='lv', processors='tokenize,pos,lemma', use_gpu=False)

    def lemmatize(self, text):
        doc = self.nlp(text)
        lemmas = [
            word.lemma for sentence in doc.sentences for word in sentence.words]
        return " ".join(lemmas)

    def get_bert_embeddings(self, texts):
        lemmatized_texts = [self.lemmatize(t) for t in texts]
        inputs = self.tokenizer(lemmatized_texts, padding=True,
                                truncation=True, return_tensors="tf")
        outputs = self.bert_model(inputs)
        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
        return embeddings.numpy()

    def train(self, user_id, liked_ids, disliked_ids):
        with self.app.app_context():
            positive = self.db.session.query(Procurement.text).filter(
               Procurement.id.in_(liked_ids)
            ).all()
           
            max_negatives = int(len(positive) * 3)

            negative = self.db.session.query(Procurement.text).filter(
                Procurement.id.in_(disliked_ids)
            ).limit(max_negatives).all()


            X = [text for (text,) in positive + negative]
            y = np.array([1] * len(positive) + [0] * len(negative))

            X_emb = self.get_bert_embeddings(X)

            model = None
            if self.db.session.query(User).filter_by(id=user_id).first().model is None:
                model = models.Sequential([
                    layers.Input(shape=(X_emb.shape[1],)),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(1, activation='sigmoid')
                ])

                model.compile(optimizer=optimizers.Adam(learning_rate=0.005),
                            loss=losses.BinaryCrossentropy(),
                            metrics=['accuracy'])
            else:
                model_path = f"/ml_models/user_model_{user_id}.h5"
                if not tf.io.gfile.exists(model_path):
                    raise ValueError(f"Modelis lietotājam {user_id} nav saglabāts")
                model = models.load_model(model_path)

            model.fit(X_emb, y, epochs=7)

            model.save(f"ml_models/user_model_{user_id}.h5")
            self.db.session.query(User).filter_by(id=user_id).update({User.model: True})
            self.db.session.commit()

    def predict(self, model, procurement_text):
        emb = self.get_bert_embeddings([procurement_text])
        pred = model.predict(emb)
        return pred[0][0]
