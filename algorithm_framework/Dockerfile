FROM python:3.8.5-slim-buster
RUN apt-get update && apt-get install -y python-opencv libopencv-dev git
COPY . /app
RUN pip install --no-cache-dir pylibjpeg pylibjpeg-libjpeg
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN cd /app/python3-gdcm && dpkg -i build_1-1_amd64.deb && apt-get install -f
RUN cp /usr/local/lib/gdcm.py /usr/local/lib/python3.8/site-packages/. && cp /usr/local/lib/gdcmswig.py /usr/local/lib/python3.8/site-packages/. && cp /usr/local/lib/_gdcmswig.so /usr/local/lib/python3.8/site-packages/. && cp /usr/local/lib/libgdcm* /usr/local/lib/python3.8/site-packages/.
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

