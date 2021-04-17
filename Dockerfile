# Use Python Official Image as Base
FROM python:3

WORKDIR /app

COPY . .

ADD main.py /

# Install required Libraries & Modules
RUN pip install -r requirements.txt

RUN [ "python", "-c", "import nltk; nltk.download('all')" ]

CMD [ "python", "./main.py" ]
