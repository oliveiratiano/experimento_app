# Use Python Official Image as Base
FROM python:3

WORKDIR /app

COPY . .

RUN apt-get update

RUN apt install gcc

RUN apt install make

# Install required Libraries & Modules
RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords

ADD main.py /

CMD [ "python", "./main.py" ]
