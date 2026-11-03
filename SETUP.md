# Block 0 setup (Windows, GTX 1050 Ti, driver 560.76 / CUDA 12.6)

Run C++ commands from the "x64 Native Tools Command Prompt for VS" so MSVC is on PATH.

## Python
```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]
python scripts\check_env.py
```

## C++
```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
build\env_check.exe
```

## Git
```bat
git init
git add .
git commit -m "Scaffold and Block 0"
```

Block 0 is done when `check_env.py` prints "PyTorch CUDA environment OK" and `env_check.exe` prints "C++ build environment OK".
