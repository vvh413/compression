#!/bin/sh

container=`docker run -dit \
    --ipc=host --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -p 8888 -v $(pwd):/work \
    vvh413/ml`
echo $container
docker port $container
docker exec -it $container sh -c "cd /work && bash"

