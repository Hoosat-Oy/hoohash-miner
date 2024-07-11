#!/bin/bash
# gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx906
for arch in gfx1032
do
  /home/tonto/RadeonDeveloperSuite/rga --O3 -s opencl -c "$arch" --OpenCLoption "-cl-finite-math-only -cl-mad-enable " -b plugins/opencl/resources/bin/hoohash-opencl.bin plugins/opencl/resources/hoohash-opencl.cl -D __FORCE_AMD_V_DOT8_U32_U4__=1 -D OPENCL_PLATFORM_AMD -D OFFLINE
done

# for arch in gfx1010
# do
#   /home/tonto/RadeonDeveloperSuite/rga --O3 -s opencl -c "$arch" --OpenCLoption "-cl-finite-math-only -cl-mad-enable " -b plugins/opencl/resources/bin/hoohash-opencl.bin plugins/opencl/resources/hoohash-opencl.cl -D OPENCL_PLATFORM_AMD
# done

# for arch in Ellesmere
# do
#   /home/tonto/RadeonDeveloperSuite/rga --O3 -s opencl -c "$arch" --OpenCLoption "-cl-finite-math-only -cl-mad-enable -target amdgcn-amd-amdpal" -b plugins/opencl/resources/bin/hoohash-opencl.bin plugins/opencl/resources/hoohash-opencl.cl -D OPENCL_PLATFORM_AMD -D PAL
# done