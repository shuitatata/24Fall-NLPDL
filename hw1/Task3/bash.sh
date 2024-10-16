#!/bin/bash

# 1. 安装requirements.txt中的依赖
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install requirements."
    exit 1
fi

# 2. 运行preprocess.py
echo "Running preprocess.py..."
python preprocess.py
if [ $? -ne 0 ]; then
    echo "Failed to run preprocess.py."
    exit 1
fi

# 3. 运行build_vocab.py
echo "Running build_vocab.py..."
python build_vocab.py
if [ $? -ne 0 ]; then
    echo "Failed to run build_vocab.py."
    exit 1
fi

# 4. 运行cbow.py
echo "Running cbow.py..."
python cbow.py
if [ $? -ne 0 ]; then
    echo "Failed to run cbow.py."
    exit 1
fi

# 5. 运行train.py
echo "Running train.py..."
python train.py
if [ $? -ne 0 ]; then
    echo "Failed to run train.py."
    exit 1
fi

# 6. 运行test.py
echo "Running test.py..."
python test.py
if [ $? -ne 0 ]; then
    echo "Failed to run test.py."
    exit 1
fi

echo "All tasks completed successfully!"
