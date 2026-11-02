# Airflow image with the training dependencies baked in (faster than
# _PIP_ADDITIONAL_REQUIREMENTS, which reinstalls them on every container start).
FROM apache/airflow:2.10.5-python3.11
COPY requirements.txt /tmp/requirements.txt
# Keep the numpy/pandas shipped with the image: letting pip upgrade numpy to 2.x
# breaks the preinstalled pandas wheels ("numpy.dtype size changed").
RUN pip freeze | grep -iE "^(numpy|pandas)==" > /tmp/pins.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/pins.txt
