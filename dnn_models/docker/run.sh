#!/bin/sh

dirname=$(pwd | xargs dirname)
dataset="/share/private/27th/hirotaka_saito/dataset/"

docker run -it \
  --privileged \
  --gpus all \
  -p 15900:5900 \
  --rm \
  --mount type=bind,source=$dirname,target=/root/vpr \
  --mount type=bind,source=$dataset,target=/root/dataset \
   --net host \
   --shm-size=40gb \
  vpr
  bash
