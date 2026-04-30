FROM supervisely/base-py-sdk:6.73.560
ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install open3d==0.18.0

RUN apt-get -y install curl
RUN apt-get update && apt -y install wireguard iproute2