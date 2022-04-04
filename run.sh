#!/bin/sh

docker run -it \
     --ipc=host --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -p 8888:8888 -v $(pwd):/work \
    ml
