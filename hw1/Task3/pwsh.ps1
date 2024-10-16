# PowerShell 脚本 run_all.ps1

# 1. 安装requirements.txt中的依赖
Write-Host "Installing requirements..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install requirements."
    exit 1
}

# 2. 运行preprocess.py
Write-Host "Running preprocess.py..."
python preprocess.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to run preprocess.py."
    exit 1
}

# 3. 运行build_vocab.py
Write-Host "Running build_vocab.py..."
python build_vocab.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to run build_vocab.py."
    exit 1
}

# 4. 运行cbow.py
Write-Host "Running cbow.py..."
python cbow.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to run cbow.py."
    exit 1
}

# 5. 运行train.py
Write-Host "Running train.py..."
python train.py
if ($LASTEXITCODE -ne 0)
