# Use an official Python base image based on Linux
FROM python:3.8

# Set the working directory in the container
WORKDIR /audio_to_midi_transcriber

# Copy the Python backend files to the container
COPY ./requirements.txt ./requirements.txt

# Create and activate the conda environment
# Assumes that environment.yml is part of the copied files
RUN pip install -r requirements.txt

COPY ./src ./src

EXPOSE 8000

# The code to run when container is started:
ENTRYPOINT ["python3", "src/app.py"]