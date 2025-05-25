# 🦙 Llama Assist

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge)](https://github.com/hacs/integration)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/M4TH1EU/llama-assist?style=for-the-badge)](./releases/)

![img.png](.github/images/llama-assist-logo-text-small.svg)

Llama Assist is a Home Assistant integration that allows you to interact with almost any LLM (Large Language Model)
using the [llama.cpp](https://github.com/ggml-org/llama.cpp) backend.

This integration creates a new **Conversation agent** in Home Assistant, which can be selected in the **Voice Assistants**
section of the Home Assistant UI and used to interact with the LLM.

> [!IMPORTANT]
> This is NOT a llama.cpp backend, it connects to an existing llama.cpp backend running on your local network or accessible
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
8. Search for "Llama Assist" and click "Install."

### Manually

1. Download the latest release from the [GitHub repository](https://github.com/M4TH1EU/llama-assist/).
2. Extract the downloaded ZIP file.
3. Copy the "custom_components/easy_computer_manager" directory to the "config/custom_components/" directory in your
   Home Assistant instance.

## Usage

To use this integration, you must setup a llama.cpp backend.
See instructions [here](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)