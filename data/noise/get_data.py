from datasets import load_dataset
import soundfile as sf
from pathlib import Path

# Load only noise split
dataset = load_dataset("FluidInference/musan", split="train")

output_dir = Path("wav")
output_dir.mkdir(exist_ok=True)

total_sec = 0
target_sec = 30 * 60  # 30 minutes

for i, sample in enumerate(dataset):
    audio = sample["audio"]
    data = audio["array"]
    sr = audio["sampling_rate"]

    duration = len(data) / sr

    if total_sec + duration > target_sec:
        break

    sf.write(output_dir / f"noise_{i}.wav", data, 16000)
    total_sec += duration

print("Collected minutes:", total_sec / 60)
