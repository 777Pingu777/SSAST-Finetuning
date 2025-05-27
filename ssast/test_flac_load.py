import soundfile as sf
import torchaudio
torchaudio.set_audio_backend("ffmpeg")

path = r'C:\Users\dfele\Documents\Personal Projects\SSAST\ssast\ASVspoof2021_DF\ASVspoof2021_DF_eval_part03\ASVspoof2021_DF_eval\flac\DF_E_4249677.flac'  # Use a known FLAC file

# Try soundfile
try:
    data, sr = sf.read(path)
    print(f"soundfile read: {data.shape}, sr={sr}")
except Exception as e:
    print(f"soundfile failed: {e}")

# Try torchaudio
try:
    wav, sr = torchaudio.load(path)
    print(f"torchaudio read: {wav.shape}, sr={sr}")
except Exception as e:
    print(f"torchaudio failed: {e}")
