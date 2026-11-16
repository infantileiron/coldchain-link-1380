from sqlalchemy import Column,Integer,String,Float,JSON,Text,Boolean,Index
from src.config.database import Base
class User(Base):
 __tablename__='users'
 id=Column(Integer,primary_key=True);email=Column(String(100),unique=True);password=Column(String(200));role=Column(String(20))
class Shipment(Base):
 __tablename__='shipments'
 id=Column(Integer,primary_key=True);code=Column(String(60),unique=True);route=Column(String(60));zone=Column(String(20));driver=Column(String(60))
class Sample(Base):
 __tablename__='samples'
 id=Column(Integer,primary_key=True);shipment_id=Column(Integer,index=True);code=Column(String(60),unique=True);exposure=Column(Float,nullable=True);excursions=Column(Integer,nullable=True);loss=Column(Float,nullable=True)
class Study(Base):
 __tablename__='studies'
 id=Column(Integer,primary_key=True);name=Column(String(100));zone=Column(String(20));feature=Column(String(30));state=Column(String(20),default='queued');result=Column(JSON,nullable=True);error=Column(Text,default='')
class Recommendation(Base):
 __tablename__='recommendations'
 id=Column(Integer,primary_key=True);study_id=Column(Integer);note=Column(Text);owner=Column(String(60));status=Column(String(20),default='draft')
class Audit(Base):
 __tablename__='audits'
 id=Column(Integer,primary_key=True);actor=Column(String(100));action=Column(String(200));created=Column(String(40))


class ModelVersion(Base):
 """一次冻结训练输入的不可变结果；被拒绝的版本也保留，供对比与审计。"""
 __tablename__='model_versions'
 id=Column(Integer,primary_key=True)
 label=Column(String(40),unique=True)
 algorithm=Column(String(60))
 state=Column(String(20),default='candidate')  # candidate / rejected
 eligible=Column(Boolean,default=False)
 snapshot_hash=Column(String(64),index=True)
 seed=Column(Integer,default=3102)
 holdout_fraction=Column(Float,default=0.2)
 snapshot=Column(JSON,nullable=True)
 normalization=Column(JSON,nullable=True)
 coefficients=Column(JSON,nullable=True)
 metrics=Column(JSON,nullable=True)
 split=Column(JSON,nullable=True)
 reason=Column(Text,default='')
 created=Column(String(40))
 created_by=Column(String(100))


class ModelState(Base):
 """单行指针：当前唯一生效版本。写入与发布记录在同一事务内提交。"""
 __tablename__='model_state'
 id=Column(Integer,primary_key=True)
 version_id=Column(Integer,nullable=True)
 deployment_id=Column(Integer,nullable=True)
 updated=Column(String(40),default='')


class ModelDeployment(Base):
 """上线/回滚记录；request_key 保证失败重试与超时重放不重复发布。"""
 __tablename__='model_deployments'
 id=Column(Integer,primary_key=True)
 version_id=Column(Integer,index=True)
 previous_version_id=Column(Integer,nullable=True)
 action=Column(String(20),default='deploy')
 request_key=Column(String(120),unique=True)
 actor=Column(String(100))
 note=Column(Text,default='')
 artifact_path=Column(String(400),default='')
 created=Column(String(40))


class ModelLock(Base):
 """可移植的发布互斥锁；过期后可被接管，避免进程崩溃留下死锁。"""
 __tablename__='model_locks'
 name=Column(String(80),primary_key=True)
 owner=Column(String(80))
 expires_at=Column(Float)


class Prediction(Base):
 """单笔预测冻结其使用的版本与参数，后续上线或回滚都不改写历史预测。"""
 __tablename__='predictions'
 id=Column(Integer,primary_key=True)
 sample_id=Column(Integer,nullable=True,index=True)
 shipment_id=Column(Integer,nullable=True)
 version_id=Column(Integer,index=True)
 version_label=Column(String(40))
 algorithm=Column(String(60))
 snapshot_hash=Column(String(64))
 input=Column(JSON)
 parameters=Column(JSON)
 output=Column(JSON)
 actual_loss=Column(Float,nullable=True)
 actor=Column(String(100))
 created=Column(String(40))
Index('ix_predictions_version_created',Prediction.version_id,Prediction.created)
