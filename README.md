# Inferencing integration onto CARPL (Binary Classification Algorithm)
![AutoAlgorithmIntegration](https://user-images.githubusercontent.com/48349718/149273334-4cd58f47-67b1-4c31-a782-207444481dd7.png)

# Use an existing model integrated using this framework
We implemented the following [chest x-ray algorithm](https://github.com/dattran2346/chestX-ray-14).
```
docker-compose up -d --build
```
This will start two docker services
* algorithm-inference-api: REST API and Scheduler service to save incoming requests to database and process them in a sequential manner.
* algorithm-mysql: Stores all the incoming requests and results

You can change the ports and environment vairables by changing the `docker-compose.yml` file.

Once the services are up and running, you can now integrate the algorithm onto your CARPL account. You can refer to [this guide](https://docs.carpl.ai/carpl/instruction-for-use/algorithm-integration-with-carpl) for the same.

# Integrate your own algorithm
You can use the existing code base in this repository. 
Requirements:
* Update the `prediction` function in the file `algorithm_framework/main.py` and your prediction function logic.
* This sample algorithm accepts a DICOM file[.dcm format]. To handle other file formats, add a preprocessing step in the `/api/upload` function.
* Outputs of the two inferencing APIs should adhere to CARPL accepted formats. Please refer to the samples below.
Sample `/api/results` response: 
```
{
   "findings": [
     {
       "name": "Finding_A",
       "probability": "19.565606117248528"
     },
     {
       "name": "Finding_B",
       "probability": "34.508439898490906"
     },
   ],
   "status": "Processed",
   "job_id": "<UNIQUE_JOB_ID>"
 }
```
Sample `/api/bbox` response: 
```
{
   "rois": [
     {
      'type': 'Freehand',
      'StudyInstanceUID': 'XXXXXXXXX',
      'SeriesInstanceUID': 'XXXXXXXXXXXX',
      'SOPInstanceUID': 'XXXXXXXXXX',
      'points': [[point_1_x, point_1_y],[point_2_x, point_2_y],[point_3_x, point_3_y], ... ,[point_n_x, point_n_y]],
       'finding_name': 'XXXX'
     },
     {
     'type': 'Rectangle',
     'StudyInstanceUID': 'XXXXXXXXX',
     'SeriesInstanceUID': 'XXXXXXXXXXXX',
     'SOPInstanceUID': 'XXXXXXXXXX',
     'points': [[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]],
     'finding_name': 'XXXX'
     }
   ],
   "status": "Processed",
   "job_id": "<UNIQUE_JOB_ID>"
 }
```

**NOTE:** If you have any concerns running this repository, please write to our team at [rohit.takhar@carpl.ai](mailto:rohit.takhar@carpl.ai).
