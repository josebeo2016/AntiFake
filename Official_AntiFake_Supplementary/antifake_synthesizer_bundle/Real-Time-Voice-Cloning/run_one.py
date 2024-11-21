import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from encoder import inference as encoder
from encoder import audio
from synthesizer.inference import Synthesizer
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
import logging
import sys

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def voice_clone(speaker_wav, text, output_path):
    ## Load the models one by one.
    ensure_default_models(Path("saved_models"))
    encoder.load_model(Path("saved_models/default/encoder.pt"))
    synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
    vocoder.load_model(Path("saved_models/default/vocoder.pt"))

    # Get the reference audio filepath
    in_fpath = Path(speaker_wav)


    preprocessed_wav = encoder.preprocess_wav(in_fpath)

    wav, wave_slices, mel_slices, _model, _device = encoder.embed_utterance_preprocess(preprocessed_wav, using_partials=True)

    wav_tensor_initial = torch.from_numpy(wav).unsqueeze(0).to(_device)

    frames_tensor = audio.wav_to_mel_spectrogram_torch(wav_tensor_initial).to(_device)
    
    embeds_list = []             
    for s in mel_slices:
        frame_tensor = frames_tensor[s].unsqueeze(0).to(_device)                
        embed = _model.forward(frame_tensor)
        embeds_list.append(embed)
    partial_embeds = torch.stack(embeds_list, dim=0)
    # The shape of raw_embed is 256
    raw_embed = torch.mean(partial_embeds, dim=0, keepdim=True)
    # Normalize the embedding, torch.Size([1, 1, 256])
    embed = raw_embed / torch.norm(raw_embed, p=2)
    
    # Convert embed to numpy array
    embed = embed.detach().cpu().numpy()[0][0]
    
    # Get the mean and sum of the embedding
    embed_mean = np.mean(embed)
    embed_sum = np.sum(embed)

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav = vocoder.infer_waveform(spec)

    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)

    # Save it on the disk
    sf.write(output_path, generated_wav.astype(np.float32), synthesizer.sample_rate)


if __name__ == '__main__':
    
    voice_clone(sys.argv[2], 
                sys.argv[1], 
                sys.argv[3])
    