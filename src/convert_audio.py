from pydub import AudioSegment
import os
import sys

# Paths
input_dir = "data/ambient/"
output_dir = "data/ambient_converted/"
os.makedirs(output_dir, exist_ok=True)

# Target format
sample_rate = 16000
duration_ms = 3000  # 3 seconds
channels = 1  # Mono

# Convert files
count = 0
for file in os.listdir(input_dir):
    if file.endswith(".wav"):
        try:
            # Load audio
            audio = AudioSegment.from_file(os.path.join(input_dir, file))
            # Convert format
            audio = audio.set_frame_rate(sample_rate).set_channels(channels)
            # Trim or pad to 3s
            if len(audio) > duration_ms:
                audio = audio[:duration_ms]
            else:
                audio = audio + AudioSegment.silent(duration=duration_ms - len(audio))
            # Save
            output_file = os.path.join(output_dir, file)
            audio.export(output_file, format="wav")
            count += 1
            print(f"Converted {file} ({count})")
        except Exception as e:
            print(f"Error converting {file}: {e}")
            continue

print(f"Converted {count} files to {output_dir}")