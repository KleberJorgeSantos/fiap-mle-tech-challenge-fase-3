# Airflow image with the training dependencies baked in (faster than
# _PIP_ADDITIONAL_REQUIREMENTS, which reinstalls them on every container start).
FROM apache/airflow:2.10.5-python3.11
COPY requirements.txt /tmp/requirements.txt
# Keep the versions Airflow ships with: upgrading numpy breaks the preinstalled pandas
# wheels ("numpy.dtype size changed"), and protobuf/OpenTelemetry are used by Airflow itself.
RUN pip freeze | grep -iE "^(numpy|pandas|protobuf|opentelemetry-[a-z-]+|cloudpickle)==" > /tmp/pins.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/pins.txt
