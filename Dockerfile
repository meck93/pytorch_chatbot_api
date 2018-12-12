FROM python:3.7-alpine

WORKDIR /usr/src/app

# COPY env.txt ./
# RUN ls -a

RUN apk --no-cache add --virtual .builddeps gcc gfortran musl-dev && pip install numpy && apk del .builddeps && rm -rf /root/.cache
RUN pip install flask
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl
RUN pip install torchvision
RUN ls -a

COPY src .
RUN ls -a

EXPOSE 5000

ENTRYPOINT ["python", "src/__init__.py"]
