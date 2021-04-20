# Use Python Official Image as Base
FROM ubuntu:latest

WORKDIR /app

COPY . .

RUN apt update

RUN apt install -y python3-pip

# Install required Libraries & Modules
RUN pip3 install -r requirements.txt

RUN python3 -m nltk.downloader stopwords

CMD [ "python3", "./main.py" ]
