FROM ubuntu:latest
LABEL authors="murad"

ENTRYPOINT ["top", "-b"]