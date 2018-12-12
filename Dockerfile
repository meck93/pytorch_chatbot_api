FROM python:3.7-alpine

WORKDIR /usr/src/app

COPY env.txt ./
RUN ls -a

RUN apk --no-cache --update-cache add musl-dev linux-headers g++
RUN pip install -r env.txt --no-cache-dir
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl --no-cache-dir
RUN pip install torchvision --no-cache-dir
RUN ls -a

COPY src .
RUN ls -a

EXPOSE 5000

ENTRYPOINT ["python", "src/__init__.py"]
