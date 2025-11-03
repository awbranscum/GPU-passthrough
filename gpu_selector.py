import os
import subprocess
import torch

def get_gpu_list():
    try:
        # Use nvidia-smi to get GPU names and indexes
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpus = [line.strip().split(", ") for line in result.stdout.strip().splitlines()]
        return [(int(idx), name) for idx, name in gpus]
    except Exception:
        # Fallback to PyTorch detection
        if torch.cuda.is_available():
            return [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
        else:
            return []

def select_gpus():
    gpus = get_gpu_list()
    if not gpus:
        print("No GPUs detected. The server will run on CPU.")
        return None

    print("\nDetected GPUs:")
    for idx, name in gpus:
        print(f"  {idx}: {name}")

    selection = input(
        "\nEnter the GPUs to use (comma-separated, in priority order): "
    ).strip()

    chosen = [s.strip() for s in selection.split(",") if s.strip().isdigit()]
    if not chosen:
        print("No valid GPUs selected. Using all available GPUs.")
        chosen = [str(idx) for idx, _ in gpus]

    # Apply CUDA_VISIBLE_DEVICES based on chosen order
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(chosen)

    # Save to file so the main server can read it
    with open("selected_gpus.txt", "w") as f:
        f.write(",".join(chosen))

    print(f"Selected GPUs (priority order): {', '.join(chosen)}")
    return chosen

if __name__ == "__main__":
    select_gpus()
