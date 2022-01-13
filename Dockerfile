FROM tiangolo/uvicorn-gunicorn:python3.8
RUN  apt-get update && apt-get install -y  \
  libdc1394-22-dev \
  python-opencv libopencv-dev
RUN pip install --no-cache-dir pylibjpeg pylibjpeg-libjpeg python-multipart pydicom numpy imutils opencv-python fastapi
COPY . /app
RUN cd /app/python3-gdcm && dpkg -i build_1-1_amd64.deb && apt-get install -f
RUN cp /usr/local/lib/gdcm.py /usr/local/lib/python3.8/site-packages/. && cp /usr/local/lib/gdcmswig.py /usr/local/lib/python3.8/site-packages/. && cp /usr/local/lib/_gdcmswig.so /usr/local/lib/python3.8/site-packages/. && cp /usr/local/lib/libgdcm* /usr/local/lib/python3.8/site-packages/.
