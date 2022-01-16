
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#SQLALCHEMY_DATABASE_URL = "sqlite:///./algorithm_integration.db"
SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://inference_user:inference_pass@algorithm-mysql:3306/inference_db"
#SQLALCHEMY_DATABASE_URL = ""postgresql://user:password@postgresserver/db""

engine = create_engine(
        SQLALCHEMY_DATABASE_URL
)

# bind – An optional Connectable, will be assigned the bind attribute on the MetaData instance.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# A simple constructor that allows initialization from kwargs.
Base = declarative_base()
