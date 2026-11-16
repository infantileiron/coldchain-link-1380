from pydantic import BaseModel,Field,model_validator
from typing import Optional,Literal,List
class Login(BaseModel):
 email:str;password:str
class ShipmentInput(BaseModel):
 code:str=Field(min_length=1,max_length=60);route:str=Field(min_length=1,max_length=60);zone:Literal['冷藏','冷冻','常温对照'];driver:str=Field(min_length=1,max_length=60)
class SampleInput(BaseModel):
 shipment_id:int;code:str=Field(min_length=1,max_length=60);exposure:Optional[float]=Field(default=None,ge=0,le=1440);excursions:Optional[int]=Field(default=None,ge=0,le=100);loss:Optional[float]=Field(default=None,ge=0,le=100)
class StudyInput(BaseModel):
 name:str=Field(min_length=1,max_length=100);zone:Literal['冷藏','冷冻','常温对照'];feature:Literal['exposure','excursions']='exposure'
class RecommendationInput(BaseModel):
 study_id:int;note:str=Field(min_length=10,max_length=2000);owner:str=Field(min_length=1,max_length=60)

class TrainInput(BaseModel):
 seed:int=3102
 holdout_fraction:float=Field(default=0.2,gt=0,lt=1)
class DeployInput(BaseModel):
 version_id:int
 request_key:Optional[str]=Field(default=None,max_length=120)
 note:str=Field(default='',max_length=500)
 expected_version_id:Optional[int]=None
class RollbackInput(BaseModel):
 target_version_id:Optional[int]=None
 request_key:Optional[str]=Field(default=None,max_length=120)
 note:str=Field(default='',max_length=500)
 expected_version_id:Optional[int]=None
class PredictRow(BaseModel):
 sample_id:Optional[int]=None
 shipment_id:Optional[int]=None
 exposure:Optional[float]=Field(default=None,ge=0,le=1440)
 excursions:Optional[int]=Field(default=None,ge=0,le=100)
class PredictInput(BaseModel):
 rows:Optional[List[PredictRow]]=None
 sample_id:Optional[int]=None
 shipment_id:Optional[int]=None
 exposure:Optional[float]=Field(default=None,ge=0,le=1440)
 excursions:Optional[int]=Field(default=None,ge=0,le=100)
 @model_validator(mode='after')
 def one_shape(self):
  if self.rows and any(x is not None for x in (self.sample_id,self.shipment_id,self.exposure,self.excursions)):
   raise ValueError('rows 与单笔字段不能同时提供')
  if self.rows is not None and len(self.rows)>100:
   raise ValueError('单次最多预测 100 条')
  return self
 def expanded(self):
  if self.rows is not None:
   return [r.model_dump() for r in self.rows]
  return [{'sample_id':self.sample_id,'shipment_id':self.shipment_id,'exposure':self.exposure,'excursions':self.excursions}]
