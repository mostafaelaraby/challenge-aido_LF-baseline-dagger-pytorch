# Definition of Submission container
ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/dt-machine-learning-base-environment:daffy-amd64

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
RUN rm -rf /workspace; mkdir /workspace
WORKDIR /workspace

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

RUN pip3 uninstall -y dataclasses

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY solution.py /workspace
COPY helpers.py /workspace
COPY models /workspace/models
COPY model.py /workspace
COPY wrappers.py /workspace

# we make the workspace our working directory


RUN python3 -c "import solution; import wrappers; import model; import helpers;"

# let's see what you've got there...
CMD python3 solution.py