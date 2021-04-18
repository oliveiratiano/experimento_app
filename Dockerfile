# Use Python Official Image as Base
FROM ubuntu:latest

WORKDIR /app

COPY . .

# Install required Libraries & Modules
RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords

ADD main.py /

CMD [ "python", "./main.py" ]
