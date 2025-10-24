FROM nvcr.io/nvidia/pytorch:24.01-py3
  
WORKDIR ~/audio-flamingo

# To have webp support, we need to install webp and then re-install pillow
RUN apt-get update && apt-get install -y webp
RUN yes | pip uninstall pillow && yes | CC="cc -mavx2" pip install -U --force-reinstall pillow-simd




RUN pip install --upgrade pip 
# Copy the local copy of llava into the docker image
COPY pyproject.toml pyproject.toml
RUN pip install -e .
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y ffmpeg netcat
RUN pip install -r requirements.txt
RUN pip install flash-attn==2.7.4.post1
RUN pip install kaldiio
RUN pip install beartype
RUN pip install opencv-python-headless==4.8.0.74
RUN pip install triton==3.1.0
