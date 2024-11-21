import torch
import torchaudio
import librosa
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import os
import logging
import sys

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def load_voice(speaker_folder, single = True):

    if single:
        paths = [speaker_folder]

    else:
        paths = [os.path.join(speaker_folder, item) for item in os.listdir(speaker_folder)]
    
    conds = []
    for cond_path in paths:
        c = load_audio(cond_path, 22050)
        conds.append(c)
        
    return conds, None



def voice_clone(tts, text, speaker_folder, out_name, single = False):
    # Load it and send it through Tortoise.
    voice_samples, conditioning_latents = load_voice(speaker_folder, single = single)
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset="fast")
    torchaudio.save(out_name, gen.squeeze(0).cpu(), 24000)
    

if __name__ == "__main__":
    
    prompts_txt = sys.argv[1]
    
    tts = TextToSpeech()
    
    voice_samples, conditioning_latents = load_voice(sys.argv[2])
    gen = tts.tts_with_preset(prompts_txt, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset="fast")
    torchaudio.save(sys.argv[3], gen.squeeze(0).cpu(), 24000)
    