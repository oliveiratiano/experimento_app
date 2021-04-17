# Use Python Official Image as Base
FROM python:3

WORKDIR /app

COPY . .

# Install required Libraries & Modules
RUN pip install -r requirements.txt

RUN sudo python -m nltk.downloader -d /usr/share/nltk_data all

ADD main.py /

CMD [ "sudo python", "./main.py" ]
