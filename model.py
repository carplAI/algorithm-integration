from sqlalchemy import Boolean, Column, Integer, String, Text
from db_connection import Base


class Details(Base):
    __tablename__ = "algo_inference"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    study_id = Column(String(100), index=True, nullable=False)
    payload = Column(Text(4294000000), index=True)
    status = Column(String(100), index=True)
    path = Column(String(256), index=True)
    #created_on = Column(String, index=True)
    #updated_on = Column(String, index=True)
