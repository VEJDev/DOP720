from sqlalchemy import Column, String, Integer, Text, LargeBinary, create_engine, DateTime
import os
from sqlalchemy.orm import declarative_base

Base = declarative_base()

def init_db():
    engine = create_engine("sqlite:///{}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")))
    Base.metadata.create_all(engine)

class Procurement(Base):

    __tablename__ = "Procurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    customer = Column(Text, nullable=False)
    deadline = Column(DateTime, nullable=False)

    def __init__(self, status, name, customer, deadline):
        self.status = status
        self.name = name
        self.customer = customer
        self.deadline = deadline

class User(Base):

    __tablename__ = "Users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(16), nullable=False, unique=True)
    email = Column(Text, nullable=False, unique=True)
    password = Column(Text, nullable=False)
    model = Column(LargeBinary, nullable=True)

    def __init__(self, username, email, password, model):
        self.username = username
        self.email = email
        self.password = password
        self.model = model