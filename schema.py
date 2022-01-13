
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, time
from fastapi import Body
from fastapi import FastAPI, File, UploadFile, Form


class DetailsBase(BaseModel):
    job_id: str
    study_id: str
    status: str
    path: str

class DetailsAdd(DetailsBase):
    payload: Optional[str] = None
    created_on: datetime = Body(None)
    updated_on: datetime = Body(None)

    class Config:
        orm_mode = True


class Detail(DetailsAdd):
    path: str
    id: int
    class Config:
        orm_mode = True

