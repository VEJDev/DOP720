from sqlalchemy import Column, String, Integer, Text, Boolean, create_engine, DateTime, ForeignKey
import os
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def init_db():
    engine = create_engine("sqlite:///{}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")))
    Base.metadata.create_all(engine)

class Procurement(Base):

    __tablename__ = "Procurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
    link = Column(Text, nullable=False)
    customer = Column(Text, nullable=False)
    deadline = Column(DateTime)

    selected_by_users_link = relationship("UserProcurement", back_populates="procurement")

    def __init__(self, status, text, link, customer, deadline):
        self.status = status
        self.text = text
        self.link = link
        self.customer = customer
        self.deadline = deadline

class User(Base):

    __tablename__ = "Users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(16), nullable=False, unique=True)
    email = Column(Text, nullable=False, unique=True)
    password = Column(Text, nullable=False)
    model = Column(Boolean, nullable=True)

    selected_procurements_link = relationship("UserProcurement", back_populates="user")

    def __init__(self, username, email, password, model):
        self.username = username
        self.email = email
        self.password = password
        self.model = model

# Kad lietotājs apmāca modeli, šeit būtu jāpiefiksē kurus iepirkumus vairs nebūtu jārāda apmācības sadaļā.
class UserProcurement(Base):
    __tablename__ = "UserProcurements"

    user_id = Column(Integer, ForeignKey("Users.id"), primary_key=True)
    procurement_id = Column(Integer, ForeignKey("Procurements.id"), primary_key=True)

    user = relationship("User", back_populates="selected_procurements_link")
    procurement = relationship("Procurement", back_populates="selected_by_users_link")

    def __init__(self, user_id, procurement_id):
        self.user_id = user_id
        self.procurement_id = procurement_id