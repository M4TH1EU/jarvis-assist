# 🦙 Llama Assist

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge)](https://github.com/hacs/integration)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/M4TH1EU/llama-assist?style=for-the-badge)](./releases/)

![img.png](.github/images/llama-assist-logo-text-small.svg)

Llama Assist is a Home Assistant integration that allows you to interact with almost any LLM (Large Language Model)
using the [llama.cpp](https://github.com/ggml-org/llama.cpp) backend.

This integration creates a new **Conversation agent** in Home Assistant, which can be selected in the **Voice Assistants
**
section of the Home Assistant UI and used to interact with the LLM.

> [!IMPORTANT]
> This is NOT a llama.cpp backend, it connects to an existing llama.cpp backend running on your local network or
> accessible
> via the internet.

## Features

- Lightweight and fast
- Easy to set up and use
- **Supports any LLMs supported by [llama.cpp](https://github.com/ggml-org/llama.cpp)**
- Supports all built-in Home Assistant **Assist** actions
- Additional actions for more advanced interactions _(COMING SOON)_

## Installation

### Via HACS

1. Install [HACS](https://hacs.xyz/) if not already installed.
2. In Home Assistant, go to "HACS" in the sidebar.
3. Click on "Integrations."
4. Click on the three dots in the top right corner and select "Custom repositories."
5. Paste the following URL in the "Repo" field: https://github.com/M4TH1EU/llama-assist
6. Select "Integration" from the "Category" dropdown.
7. Click "Add."
8. Search for **"Llama Assist"** and click "Install."

### Manually

1. Download the latest release from the [GitHub repository](https://github.com/M4TH1EU/llama-assist/).
2. Extract the downloaded ZIP file.
3. Copy the `custom_components/easy_computer_manager` directory to the `config/custom_components/` directory in your
   Home Assistant instance.

## Usage

Go to Settings -> Devices & Services -> Add Integration and search for "Llama Assist".
Fill in the required fields:
- **URL**: The URL of the [llama.cpp HTTP backend](https://github.com/ggml-org/llama.cpp/tree/master/tools/server). This can be a local IP address or a public URL. (ex: http://localhost:8080)


To use this integration, you must setup a [llama.cpp HTTP backend](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
See instructions [here](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)

### Build llama.cpp

Official documentation can be found [here](https://github.com/ggml-org/llama.cpp/tree/master).

_You might be able to use pre-built executable which can be found in the releases of llama.cpp repository_

> [!NOTE]
> Theses scripts are provided as examples that worked for me, you may need to adapt them to your system.  
> **Please do NOT open issues related to building llama.cpp, this is not the purpose of this repository.**  
> If you have issues, please open an issue on the llama.cpp repository.

<details>
<summary>Intel CPUs (oneAPI)</summary>

This script is for building llama.cpp with Intel oneAPI compiler.

```bash
#!/bin/bash
sudo apt install intel-oneapi-base-toolkit # Required to build llama.cpp for Intel CPUs

rm -Rf llama.cpp
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git llama.cpp

source /opt/intel/oneapi/setvars.sh # You can skip this step if  in oneapi-basekit docker image, only required for manual installation
cd llama.cpp/
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
cmake --build build --config Release
```

*The executable will be in `llama.cpp/build/bin/llama-server`*
</details>

<details>
  <summary>AMD GPU (ROCM)</summary>

This script is for building llama.cpp with AMD ROCM compiler, this has been tested on Fedora 42 with ROCM 6.3.1

```bash
#!/bin/bash

# This script compiles llamacpp for ROCM under fedora (tested on 42), must have all 'rocm*'
# packages installed along with hipblas and other stuff...
# sudo dnf install 'rocm*' 'hipblaslt' 'hipblas-*' rocblas-devel make gcc cmake libcurl-devel

rm -rf sources/
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git sources

cd sources/

MAX_THREADS=8

# Automatically detect HIP configuration paths
HIPCXX=$(hipconfig -l)/clang
HIP_PATH=$(hipconfig -R)
HIP_VISIBLE_DEVICES=$(hipconfig -R)

# Ensure hipconfig is successful
if [[ -z "$HIP_PATH" ]]; then
  echo "Error: Unable to detect HIP_PATH. Ensure HIP is correctly installed."
  exit 1
fi

# Automatically detect AMDGPU_TARGETS
AMDGPU_TARGET=$(rocminfo | grep gfx | head -1 | awk '{print $2}')
if [[ -z "$AMDGPU_TARGET" ]]; then
  echo "Error: Unable to detect AMDGPU target using rocminfo."
  exit 1
fi

# Find HIP device library path
HIP_DEVICE_LIB_PATH=$(find "${HIP_PATH}" -name "oclc_abi_version_400.bc" -exec dirname {} \; | head -n 1)
if [[ -z "$HIP_DEVICE_LIB_PATH" ]]; then
  echo "Error: Unable to find oclc_abi_version_400.bc under HIP_PATH."
  exit 1
fi

# Export necessary paths
export HIPCXX
export HIP_PATH
export HIP_VISIBLE_DEVICES
export HIP_DEVICE_LIB_PATH
export DEVICE_LIB_PATH=$HIP_DEVICE_LIB_PATH
export ROCM_PATH=/usr/

# Automatically detect clang and clang++ if installed
CLANG_C_COMPILER=$(which clang)
CLANG_CXX_COMPILER=$(which clang++)

# Ensure clang is detected
if [[ ! -x "$CLANG_C_COMPILER" ]]; then
  echo "Error: clang compiler not found."
  exit 1
fi
if [[ ! -x "$CLANG_CXX_COMPILER" ]]; then
  echo "Error: clang++ compiler not found."
  exit 1
fi

# Clean build directory
rm -rf build/*
# Run cmake with dynamically detected variables
cmake -S . -B build \
  -DGGML_HIPBLAS=ON \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGET" \
  -DCMAKE_C_COMPILER="$CLANG_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$CLANG_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$ROCM_PATH
  
# Build the project
cmake --build build --config Release -- -j $MAX_THREADS
```

*The executables will be in `sources/build/bin/llama-server`*
</details>

### Other backends?
While untested, any OpenAI-API compatible backend should work with this integration.