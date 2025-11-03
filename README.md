# Ollama-Compatible GPU Server

A lightweight FastAPI server that provides an Ollama-style API for running causal language models on local GPUs. The project loads models from Hugging Face (transformers) and automatically selects available GPUs based on a user-prioritized list. It exposes simple endpoints to list loaded models, generate text, and run chat-style conversations.

---

## Key Features

* Ollama-compatible REST endpoints for generation and chat.
* Automatic model caching to avoid reloading the same model repeatedly.
* GPU selection and prioritization via a simple text file.
* Graceful fallback to CPU if no CUDA GPUs are available.
* Designed for local, development, or research use with models from Hugging Face.

---

## Repository Layout (files you will find)

* ollama_gpu_server.py — main FastAPI server implementation
* gpu_selector.py (optional) — utility to create or update the GPU priority file
* selected_gpus.txt — plain-text file that stores GPU indices in priority order (created/edited by gpu_selector.py)
* setup_ollama_gpu_server.bat — Windows batch script to create a virtual environment and install/update dependencies
* install_dependencies.bat — alternate batch installer (without venv)
* README.md — this file

---

## Requirements (high level)

* Python 3.9 or newer
* NVIDIA GPU drivers and CUDA (optional — server will run on CPU if CUDA is unavailable)
* Internet connection for downloading models from Hugging Face (unless models are cached locally)
* Adequate disk space for model checkpoints

---

## Installation (conceptual steps)

1. Ensure a suitable Python version is installed and available on your PATH.
2. Create and activate a virtual environment to isolate project dependencies.
3. Use the provided Windows batch installer to upgrade pip tools and install Python packages: FastAPI, Uvicorn, Pydantic, Transformers, and an appropriate PyTorch build (GPU-enabled if you have CUDA).
4. Confirm that packages import without errors and that torch recognizes your GPU(s) if present.

The repository includes a ready-to-run batch script that automates these steps for Windows users.

---

## Configuration

* GPU selection: To control which GPUs the server uses and the order of preference, create or edit selected_gpus.txt with a comma-separated list of GPU indices (for example: 0,1,2). When present, the server will set CUDA_VISIBLE_DEVICES accordingly and attempt to allocate models on the first available device in that order.
* Model names: When calling the API, specify a Hugging Face model identifier (for example: a local path or the HF model repo name). The server will download and cache the model on first use.

---

## API Endpoints (behavioral description)

* GET /
  Returns a status object indicating the server is running.

* GET /api/tags
  Returns a list of currently loaded models. Each entry contains the model name.

* POST /api/generate
  Accepts a JSON object with at least two fields: model (string) and prompt (string). The server loads or reuses the requested model, tokenizes the prompt, generates tokens, decodes them, and returns the generated text along with metadata like model name and timestamp. Optional settings can be passed in an options dictionary (for example, maximum new tokens).

* POST /api/chat
  Accepts a JSON object with: model (string), messages (an array of message objects consisting of role and content), and optional options. The server concatenates the conversation into a single string, tokenizes, generates a response, and returns it in an assistant message object.

* GET /api/show?model=MODEL_NAME
  Returns information about the requested model if it is loaded: model name, device in use (cpu or cuda:X), and an approximate parameter count.

---

## Usage Notes (practical guidance without commands)

* Running the server: After dependencies are installed and the virtual environment is activated, start the FastAPI application via a Python interpreter. The server prints a ready message and a local URL (for example, a localhost address and port) where you can test endpoints.
* Model loading: The first request to a new model may take time while weights are downloaded and loaded into memory. Subsequent requests will be faster due to caching.
* Memory considerations: Large models require significant GPU memory. If a model fails to load on a chosen GPU, the server attempts to find another usable device in the priority list. If none are usable, it falls back to the CPU.
* Streaming: The server supports a stream flag in request schemas but the provided implementation returns the full response after generation. If you need streaming token-by-token output for low-latency applications, consider extending the generate/chat handlers.

---

## Troubleshooting

* If no GPU is detected, confirm that NVIDIA drivers and CUDA are installed and that torch was installed with CUDA support matching your system.
* If a model fails to load due to out-of-memory errors, try a smaller model, reduce batch sizes, or change the GPU priority file to pick a device with more free memory.
* If dependency installation fails, check your Python version and that your virtual environment is active. Verify internet connectivity for package downloads.
* If the server prints an error about selected_gpus.txt, run the included GPU selector utility or create the file manually with a comma-separated list of device indices.

---

## Security and Privacy

* This server is intended for local or trusted network use. If exposing to untrusted networks, add proper authentication, rate limiting, and HTTPS.
* Model outputs are generated by third-party model weights (Hugging Face). Be mindful of licensing and the potential for biased, hallucinated, or unsafe outputs depending on the model used.
* Do not serve sensitive data through the server unless you have audited the models and your deployment for privacy and security requirements.

---

## Extensibility

* Add support for more generation options (temperature, top_k, top_p, repetition penalty, stopping criteria) by extending the options dictionary handling in the generate/chat endpoints.
* Implement token streaming using server-sent events or websockets for interactive applications.
* Add model loading strategies such as gradient checkpointing, quantized weights (8-bit/4-bit), or device mapping to better support large models on constrained GPUs.

---

## Contributing

* Contributions are welcome. Suggested workflow:

  * Create an issue describing the bug or enhancement.
  * Fork the repository and implement changes in a feature branch.
  * Open a pull request with a clear description of what changed and why.

---

## License & Acknowledgements

* Include a license appropriate for your project (for example, MIT, Apache 2.0, or another). Make sure you also follow the licenses of the models and third-party libraries you use.
* This project uses Hugging Face Transformers and PyTorch. Thanks to those communities for providing the underlying model and tensor libraries.

---

## Example Requests (described)

* To generate text, send a POST request to the generate endpoint with a JSON payload containing the model name and prompt. Include optional generation settings in an options dictionary such as the maximum number of new tokens to generate.
* To run a chat-style conversation, send a POST request to the chat endpoint with the model and an array of messages, each containing a role and content. The server will return an assistant message with the generated reply.
