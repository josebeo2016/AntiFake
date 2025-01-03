import os
import sys
import shutil
from asv_redimnet import asv_score
import librosa

if __name__ == '__main__':
    # Input text
    # input_text = input("Enter the text to generate: ")
    input_text = 'To convert the audio file to the acceptable format of AntiFake, please use the following'

    # Load source speaker
    source_speaker = sys.argv[1]
    config_path = sys.argv[2]
    speaker_name = os.path.splitext(os.path.basename(source_speaker))[0]

    # Create a new directory, remove if it already exists
    print("Creating folders...")
    shutil.rmtree(f"samples/{speaker_name}", ignore_errors=True)
    os.makedirs(f"samples/{speaker_name}/source", exist_ok=True)
    os.makedirs(f"samples/{speaker_name}/protected", exist_ok=True)

    # Copy the source speaker to the source directory
    shutil.copy(source_speaker, f"samples/{speaker_name}/source/source.wav")

    # Add watermark to the source speaker
    print("Adding watermark to the source speaker...")
    os.system(f"python run.py samples/{speaker_name}/source/source.wav samples/{speaker_name}/protected/protected.wav logs/{speaker_name} {config_path}")

    # Generate the audio
    print("Generating audio using Tortoise-TTS...")
    os.chdir("Official_AntiFake_Supplementary/antifake_synthesizer_bundle/tortoise-tts")

    # Generate TTS audio from source speaker
    os.system(f"python run_one.py \"{input_text}\" ../../../samples/{speaker_name}/source/source.wav ../../../samples/{speaker_name}/source/source_tor.wav")

    # Generate TTS audio from protected speaker
    os.system(f"python run_one.py \"{input_text}\" ../../../samples/{speaker_name}/protected/protected.wav ../../../samples/{speaker_name}/protected/protected_tor.wav")

    print("Audio generated successfully!")
    print(f"Please check the samples/{speaker_name} directory for the generated audio files.")
    
        # calculate the ASV score
    print("Calculating the ASV score...")
    os.chdir("../../..")
    source_wav, _ = librosa.load(f"samples/{speaker_name}/source/source.wav", sr=16000, mono=True)
    protected_wav, _ = librosa.load(f"samples/{speaker_name}/protected/protected.wav", sr=16000, mono=True)
    source_tor_wav, _ = librosa.load(f"samples/{speaker_name}/source/source_tor.wav", sr=16000, mono=True)
    protected_tor_wav, _ = librosa.load(f"samples/{speaker_name}/protected/protected_tor.wav", sr=16000, mono=True)
    print("Score source vs TTS Tor from source: ", asv_score(source_wav, source_tor_wav))
    
    print("Score source vs protected: ", asv_score(source_wav, protected_wav))
    print("Score source vs TTS Tor from protected: ", asv_score(source_wav, protected_tor_wav))
