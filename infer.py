import os
import sys

if __name__ == '__main__':
    # input text
    input_text = input("Enter the text to generate: ")
    
    # load source speaker
    source_speaker = sys.argv[1]
    gpu_num = sys.argv[2]
    speaker_name = source_speaker.split('/')[-1].split('.')[0]
    
    # create a new directory, remove if it already exists
    print("Creating folders...")
    os.system(f"rm -rf samples/{speaker_name}")
    os.system(f"mkdir samples/{speaker_name}")
    os.system(f"mkdir samples/{speaker_name}/source")
    os.system(f"mkdir samples/{speaker_name}/protected")
    
    # copy the source speaker to the source directory
    os.system(f"cp {source_speaker} samples/{speaker_name}/source/source.wav")
    
    # Add watermark to the source speaker
    print("Adding watermark to the source speaker...")
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_num} python run.py samples/{speaker_name}/source/source.wav samples/{speaker_name}/protected/protected.wav")
    
    # generate the audio #
    # go to the tortoise-tts directory
    print("Generating audio using Tortoise-TTS...")
    os.chdir("Official_AntiFake_Supplementary/antifake_synthesizer_bundle/tortoise-tts")
    # generate TTS audio from source speaker
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_num} python run_one.py \"{input_text}\"  ../../samples/{speaker_name}/source/source.wav ../../samples/{speaker_name}/source/source_tor.wav")
    # generate TTS audio from protected speaker
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_num} python run_one.py \"{input_text}\"  ../../samples/{speaker_name}/protected/protected.wav ../../samples/{speaker_name}/protected/protected_tor.wav")
    
    print("Audio generated successfully!")
    print(f"Please check the samples/{speaker_name} directory for the generated audio files.")