# Use Python Official Image as Base
FROM python:3

WORKDIR /app

COPY . .

# Install required Libraries & Modules
RUN pip install -r requirements.txt

# Download NLTK DATA
RUN python -m nltk.downloader all
