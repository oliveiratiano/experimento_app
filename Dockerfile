# Use Python Official Image as Base
FROM ubuntu:latest

WORKDIR /app

COPY . .

# Install required Libraries & Modules
RUN pip3 install -r requirements.txt

RUN python3 -m nltk.downloader stopwords

ADD main.py /

CMD [ "python", "./main.py" ]
