FROM python:3.9-slim-bookworm

ENV WORKDIR /var/project
WORKDIR ${WORKDIR}

# update debian
RUN apt-get update && apt-get upgrade

# Upgrade pip
RUN pip install --upgrade pip

# Install Dependencies
COPY ./requirements.txt /var/project/requirements.txt
RUN pip install -r requirements.txt

# Copy Project
#COPY . /var/project

EXPOSE 80

# Prevent Container from exiting
ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
