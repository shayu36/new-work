#!/bin/bash
set -e

# ========== 环境变量 ==========
eval "$(/data/anaconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate /data/jxy/projects/env

export CUDA_HOME=/data/jxy/projects/env/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MMCV_WITH_OPS=1
export FORCE_CUDA=1

cd /data/jxy/projects

echo "========================================"
echo "Step 1/6: 安装 mmcv-full==1.7.0 (编译约10-30分钟)"
echo "========================================"
pip install mmcv-full==1.7.0 --no-build-isolation

echo "========================================"
echo "Step 2/6: 安装 mmdet==2.28.0"
echo "========================================"
pip install mmdet==2.28.0

echo "========================================"
echo "Step 3/6: 安装 numba==0.53.0 + llvmlite==0.36.0"
echo "========================================"
pip install llvmlite==0.36.0 numba==0.53.0

echo "========================================"
echo "Step 4/6: 安装 torch-scatter==2.1.2"
echo "========================================"
pip install torch-scatter==2.1.2

echo "========================================"
echo "Step 5/6: 安装项目本体 (pip install -e .)"
echo "========================================"
pip install -e . --no-build-isolation

echo "========================================"
echo "Step 6/6: 编译 csrc CUDA 扩展"
echo "========================================"
cd /data/jxy/projects/mmdet3d/models/sparsedetectors/csrc
python setup.py build_ext --inplace
cd /data/jxy/projects

echo "========================================"
echo "全部完成！验证安装..."
echo "========================================"
python -c "
import torch
import mmcv
import mmdet
print(f'torch:   {torch.__version__}')
print(f'mmcv:    {mmcv.__version__}')
print(f'mmdet:   {mmdet.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
print(f'GPUs:    {torch.cuda.device_count()}')
print('所有依赖安装成功!')
"
