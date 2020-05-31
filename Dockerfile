FROM python:3.7
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN download.sh
CMD python main.py
