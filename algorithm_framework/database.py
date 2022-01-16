from sqlalchemy.orm import Session
import schema
from log import send_log
import datetime
import time
import model

def get_jobs_by_job_id(db: Session, id: str):
    return db.query(model.Details).filter(model.Details.job_id == id).order_by(model.Details.id.desc()).first()


def get_jobs_by_study_instance_id(db: Session, study_instance_id: str):
    return db.query(model.Details).filter(model.Details.study_id == study_instance_id).order_by(model.Details.id.desc()).first()


def get_pending_jobs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(model.Details).filter(model.Details.status=='Queued').offset(skip).limit(limit).all()

def update_job_status(db: Session, job_id: str, status: str):
    db.query(model.Details).filter(model.Details.job_id==job_id).update({model.Details.status: status})
    db.commit()

def update_response(db: Session, job_id: str, payload: str, roi: str, status: str):
    db.query(model.Details).filter(model.Details.job_id==job_id).update({model.Details.status: status, model.Details.payload: payload, model.Details.roi: roi})
    db.commit()


def add_job_to_db(db: Session, inference_data: schema.DetailsAdd):
    try:
        mv_details = model.Details(
            job_id=inference_data.job_id,
            study_id=inference_data.study_id,
            payload=inference_data.payload,
            status='Queued',
            path=inference_data.path
        )

        db.add(mv_details)
        db.commit()
        send_log(mv_details.job_id, '', "saved into Database", '', 5, False)
        return "Success"
    except Exception as e:
        print(e)
        send_log(inference_data.job_id, '', "An Error occurred while saving data into Database",
                 '', 5, False)
        return "Failed"
