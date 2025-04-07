import numpy as np
import aubio
import soundfile as sf
import sounddevice as sd
from pythonosc import udp_client
import time

BUFFER_SIZE = 256  # Must be a power of 2. Change this to increase/decrease beat sensitivity
SAMPLERATE = 44100  # Standard sample rate. Change to samplerate of file if a warning is seen on running the program
OSC_ADDRESS = "127.0.0.1"  # Localhost
OSC_PORT = 5001  # OSC listening port
AUDIO_FILE = "music.ogg"  #Replace with path of any other music file you would like to run

client = udp_client.SimpleUDPClient(OSC_ADDRESS, OSC_PORT)

audio_data, file_samplerate = sf.read(AUDIO_FILE, dtype="float32")

if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1) 

if file_samplerate != SAMPLERATE:
    print(f"Warning: File samplerate ({file_samplerate}) differs from expected ({SAMPLERATE}).")

beat_detector = aubio.tempo("default", BUFFER_SIZE, BUFFER_SIZE, file_samplerate)

frame_index = 0

def callback(outdata, frames, time, status):
    global frame_index

    if frame_index + frames > len(audio_data):
        raise sd.CallbackStop 

    outdata[:] = audio_data[frame_index: frame_index + frames].reshape(-1, 1)

    for i in range(0, frames, BUFFER_SIZE):
        frame = audio_data[frame_index + i: frame_index + i + BUFFER_SIZE]

        if len(frame) < BUFFER_SIZE:
            frame = np.pad(frame, (0, BUFFER_SIZE - len(frame)), 'constant')

        if beat_detector(frame):
            timestamp = (frame_index + i) / file_samplerate
            print(f"Beat detected at {timestamp:.2f} seconds!" )
            client.send_message("/beat", 1) 

    frame_index += frames

print("Playing audio and detecting beats...")
with sd.OutputStream(samplerate=file_samplerate, channels=1, callback=callback):
    sd.sleep(int(len(audio_data) / file_samplerate * 1000))

print("Playback and beat detection complete.")
