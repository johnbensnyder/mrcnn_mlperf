#!/bin/bash

# Train mask rcnn using Nvidia's MLPerf submission on a single P3dn

# Download COCO data from S3

mkdir -p /home/ubuntu/data
aws s3 cp s3://jbsnyder-sagemaker/data/coco/compressed_for_sagemaker/coco.tar /home/ubuntu/data
pushd /home/ubuntu/data
tar -xf coco.tar
popd
sudo chmod -R 644 /home/ubuntu/data

# cloen mlperf repo

git clone https://github.com/mlperf/training_results_v0.7
cd training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/

# build docker image
docker build -t mrcnn .

# launch training
# these are the DGX-1 settings and should train
# to convergence on a p3dn in a little under 2.5 hours
CONTAINER_NAME=mlperf_training
IMAGE_NAME=mrcnn
BASE_LR=0.06
MAX_ITER=40000
WARMUP_FACTOR=0.000096
WARMUP_ITERS=625
STEPS="\"(24000,32000)\""
TRAIN_IMS_PER_BATCH=48
TEST_IMS_PER_BATCH=8
FPN_POST_NMS_TOP_N_TRAIN=6000
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=20
NPROC_PER_NODE=8

docker run -it --rm --gpus all --name mlperf_training \
	--net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
	--ulimit=stack=67108864 --ulimit=memlock=-1 \
	-v /home/ubuntu/data:/workspace/object_detection/datasets mrcnn /bin/bash -c \
		"python -u -m bind_launch --nsockets_per_node=${NSOCKETS_PER_NODE} \
			--ncores_per_socket=${NCORES_PER_SOCKET} --nproc_per_node=${NPROC_PER_NODE} \
			tools/train_mlperf.py --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
			DTYPE 'float16' \
			PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog.py' \
			DISABLE_REDUCED_LOGGING True \
			SOLVER.BASE_LR ${BASE_LR} \
			SOLVER.MAX_ITER ${MAX_ITER} \
			SOLVER.WARMUP_FACTOR ${WARMUP_FACTOR} \
			SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
			SOLVER.WARMUP_METHOD mlperf_linear \
			SOLVER.STEPS ${STEPS} \
			SOLVER.IMS_PER_BATCH ${TRAIN_IMS_PER_BATCH} \
			TEST.IMS_PER_BATCH ${TEST_IMS_PER_BATCH} \
			MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN ${FPN_POST_NMS_TOP_N_TRAIN} \
			NHWC True"

