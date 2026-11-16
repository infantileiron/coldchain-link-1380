import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
_url=os.environ['DATABASE_URL']
engine=create_engine(_url,pool_pre_ping=True,connect_args={'check_same_thread':False} if _url.startswith('sqlite') else {})
Session=sessionmaker(bind=engine,expire_on_commit=False)
Base=declarative_base()
def db():
 with Session() as s: yield s
