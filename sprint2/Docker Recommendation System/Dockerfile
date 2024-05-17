FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt


RUN pip install --upgrade pip 

RUN pip install -r /code/requirements.txt 

COPY ./app /code/app 

EXPOSE 80

CMD ["fastapi", "--version"]
# CMD ["fastapi", "run", "app/app.py", "--port", "80"]