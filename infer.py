import os
import sys
from asv_redimnet import asv_score
import librosa
if __name__ == '__main__':
    # input text
    # input_text = input("Enter the text to generate: ")
    input_text = 'To convert the audio file to the acceptable format of AntiFake, please use the following'
    
    
    # load source speaker
    source_speaker = sys.argv[1]
    config_path = sys.argv[2]

    speaker_name = source_speaker.split('/')[-1].split('.')[0]
    
    # create a new directory, remove if it already exists
    print("Creating folders...")
    os.system(f"rm -rf samples/{speaker_name}")
    os.system(f"mkdir samples/{speaker_name}")
    os.system(f"mkdir samples/{speaker_name}/source")
    os.system(f"mkdir samples/{speaker_name}/protected")
    
    # copy the source speaker to the source directory
    # os.system(f"cp {source_speaker} samples/{speaker_name}/source/source.wav")
    os.system(f"ffmpeg -i {source_speaker} -acodec pcm_s16le -ac 1 -ar 16000 -ab 256k samples/{speaker_name}/source/source.wav")
    
    # Add watermark to the source speaker
    print("Adding watermark to the source speaker...")
    os.system(f"python run.py samples/{speaker_name}/source/source.wav samples/{speaker_name}/protected/protected.wav logs/{speaker_name} {config_path}")
    
    # generate the audio #
    # go to the tortoise-tts directory
    print("Generating audio using Tortoise-TTS...")
    os.chdir("Official_AntiFake_Supplementary/antifake_synthesizer_bundle/tortoise-tts")
    # generate TTS audio from source speaker
    os.system(f"python run_one.py \"{input_text}\"  ../../../samples/{speaker_name}/source/source.wav ../../../samples/{speaker_name}/source/source_tor.wav")
    # generate TTS audio from protected speaker
    os.system(f"python run_one.py \"{input_text}\"  ../../../samples/{speaker_name}/protected/protected.wav ../../../samples/{speaker_name}/protected/protected_tor.wav")
    
    print("Audio generated successfully!")
    print(f"Please check the samples/{speaker_name} directory for the generated audio files.")
    
    # calculate the ASV score
    print("Calculating the ASV score...")
    os.chdir("../../..")
    source_wav, sr = librosa.load(f"samples/{speaker_name}/source/source.wav", sr=16000, mono=True)
    protected_wav, sr = librosa.load(f"samples/{speaker_name}/protected/protected.wav", sr=16000, mono=True)
    source_tor_wav, sr = librosa.load(f"samples/{speaker_name}/source/source_tor.wav", sr=16000, mono=True)
    protected_tor_wav, sr = librosa.load(f"samples/{speaker_name}/protected/protected_tor.wav", sr=16000, mono=True)
    print("Score source vs TTS Tor from source: ", asv_score(source_wav, source_tor_wav))
    
    print("Score source vs protected: ", asv_score(source_wav, protected_wav))
    print("Score source vs TTS Tor from protected: ", asv_score(source_wav, protected_tor_wav))
    