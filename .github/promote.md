# 🦙 Llama Assist – Local LLMs in Home Assistant via llama.cpp 
 
Hey everyone! I’m excited to introduce **Llama Assist**, a new custom Home Assistant integration that lets you control your smart home with your **own locally hosted LLM** – especially using the blazing fast [`llama.cpp`](https://github.com/ggml-org/llama.cpp) HTTP backend.

*This project is still in very early stages and any testing is welcomed.*

🔗 GitHub: [https://github.com/M4TH1EU/llama-assist](https://github.com/M4TH1EU/llama-assist)

# What is it?

Llama Assist is a Conversation agent for Home Assistant that integrates seamlessly into the Assist ecosystem. It enables natural language interaction with your smart home using your own LLM backend, without needing any cloud services.

The integration is designed first and foremost to work with [`llama.cpp`](https://github.com/ggml-org/llama.cpp)'s **native HTTP server** — meaning you can run efficient, quantized models directly on your own CPU or GPU hardware. But it should also works with any backend compatible with the OpenAI API, as long as it supports tool/function calling.

# Key Features

* Works great with `llama.cpp`'s HTTP server (fast, private, local)
* Integrates into the Home Assistant Assist system
* Supports all built-in Assist actions + tool/function calling
* **🧠 Optional embeddings support for much faster, "cheaper" requests**
* Low hardware requirements — runs even on CPUs

# Performance

With embeddings enabled, Llama Assist can **cut response time by half and reduce token usage by two-thirds.**

Example:

    User: Hi Jarvis!
    Assistant: Hello! How can I assist you today?
    User: Add strawberries to my shopping list.
    ToolCall: HassShoppingListAddItem
    Assistant: Strawberries have been added to your shopping list.

➡️ With embeddings: 12s total, \~2000 tokens  
➡️ Without embeddings: 28s total, \~6200 tokens

🔗 GitHub: [https://github.com/M4TH1EU/llama-assist](https://github.com/M4TH1EU/llama-assist)