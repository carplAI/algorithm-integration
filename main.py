import sys, os
from fastapi import FastAPI, HTTPException, File, Form, UploadFile, Depends, Request
from fastapi.responses import JSONResponse
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/cxr_caring")
from cxr_caring.src.cxr_inference import inference_xray
version = f"{sys.version_info.major}.{sys.version_info.minor}"

app = FastAPI()
import os
import uuid
import constants
import glob, shutil
import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import database
from db_connection import SessionLocal, engine
from schema import *
import model

model.Base.metadata.create_all(bind=engine)

logger = logging.getLogger(__name__)

class Properties(BaseModel):
    job_id: str = None
    study_id: str = None
    payload: str = None
    status: str = None
    path: str = None

def verify_token(req: Request):
    token = req.headers["Authorization"]
    valid = False
    if token == 'ABC':
        valid = True
    return valid

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@app.post('/api/upload/')
async def data_upload(file: UploadFile = File(...), authorized: bool = Depends(verify_token)):
    if not authorized:
        return HTTPException(
            status_code=401,
            detail="Unauthorized"
        )
    try:
        ext = file.filename.split('.')[-1]
        if ext not in constants.ACCEPTABLE_EXTS:
            message = "Invalid file format. Accepted formats: %s " % ', '.join(constants.ACCEPTABLE_EXTS)
            return JSONResponse({"status": "fail", "message": message})

        fullname = "%s.%s" % (uuid.uuid4(), ext)
        fullpath = os.path.join(constants.TMP_FOLDER, fullname)
        try:
            with open(fullpath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except:
            return HTTPException(status_code=500, detail='Some exception occurred')
        finally:
            file.file.close()
        x = Properties(id=id, job_id=fullname, study_id='', payload='', status='Queued', path=fullpath)
        db = SessionLocal()
        db_result = database.add_job_to_db(db=db, inference_data=x)
        return JSONResponse({"status": db_result})
    except Exception as e:
        print("There is some error at ", e)
        return HTTPException(status_code=500, detail='Some exception occurred')


@app.post('/api/results/')
async def fetch_response(study_instance_uid: str = Form(...), authorized: bool = Depends(verify_token)):
    pass


@app.post('/api/bbox/')
async def fetch_response(study_instance_uid: str = Form(...), authorized: bool = Depends(verify_token)):
    pass



def prediction(job_id, path, results_path, jpg_path):
    '''
    Change this function to include your inferencing logic
    '''
    response, results_ = inference_xray(path, results_path, jpg_path)
    print(response, results_)


async def process_study(queue):
    loop = asyncio.get_event_loop()
    while True:
        job = await queue.get()
        logger.info(f"I'm going to process {job.job_id}...")
        await loop.run_in_executor(None, prediction, job.job_id, job.path, os.path.join(constants.RESULTS_FOLDER,job.study_id),os.path.join(constants.RESULTS_FOLDER,job.study_id,constants.JPG_FOLDER))
        queue.task_done()
        db = SessionLocal()
        database.update_job_status(db=db, job_id=job_id, status='Completed')
        logger.info(f"Job {job.job_id} has been processed.")


class SchedulerService:
    async def check_db_new_cases(self):
        logger.info("Checking database for new studies to process.")
        db = SessionLocal()
        all_jobs = database.get_pending_jobs(db=db)
        print(all_jobs)
        for job in all_jobs:
            db = SessionLocal()
            database.update_job_status(db=db, job_id=job.job_id, status='Proceessing')
            await asyncio.sleep(2)
            logger.info(f"Queueing {job.job_id}.")
            self.queue.put_nowait(job)
        logger.info("Checked all the studies in the DB.")

        # Create 3 workers to process the queue concurrently.
        tasks = []
        for i in range(1):
            task = asyncio.create_task(process_study(self.queue))
            tasks.append(task)
        await self.queue.join()

        # Cancel our worker tasks.
        for task in tasks:
            task.cancel()
        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*tasks, return_exceptions=True)

    def start(self):
        if not os.path.exists(constants.TMP_FOLDER):
            os.mkdir(constants.TMP_FOLDER)
        logger.info("Starting scheduler service.")
        self.queue = asyncio.Queue()
        self.sch = AsyncIOScheduler()
        self.sch.start()
        self.sch.add_job(self.check_db_new_cases, 'interval', seconds=5,
                         # Using max_instances=1 guarantees that only one job
                         # runs at the same time (in this event loop).
                         max_instances=1)

@app.on_event("startup")
async def run_scheduler():
    sch_srv = SchedulerService()
    sch_srv.start()
