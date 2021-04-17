# Use Python Official Image as Base
FROM python:3

# Install required Libraries & Modules
RUN pip install -r requirements.txt

# Download NLTK DATA
RUN python -m nltk.downloader all
