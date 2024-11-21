import torch
import torchaudio
from pathlib import Path
import sys
from sklearn.mixture import GaussianMixture
import numpy as np

sys.path.insert(0, "./AntiFake/rtvc")
from encoder import inference as encoder

def main():
    
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
     
    encoder.load_model(Path('./saved_models/default/encoder.pt'))
    
    in_fpath = Path(filename1.replace("\"", "").replace("\'", ""))
    preprocessed_wav = encoder.preprocess_wav(in_fpath, 16000)
    wav, _, mel_slices, rtvc_encoder_model, _ = encoder.embed_utterance_preprocess(preprocessed_wav, using_partials=True)
    wav1 = torch.from_numpy(wav).unsqueeze(0)
     
     
    in_fpath = Path(filename2.replace("\"", "").replace("\'", ""))
    preprocessed_wav = encoder.preprocess_wav(in_fpath, 16000)
    wav, _, mel_slices, rtvc_encoder_model, _ = encoder.embed_utterance_preprocess(preprocessed_wav, using_partials=True)
    wav2 = torch.from_numpy(wav).unsqueeze(0)
    
    print(wav1.shape)
    print(wav2.shape)
    assert wav1.size() == wav2.size(), "Waveforms must have the same length"

    # Calculate the difference between the waveforms
    diff_waveform = wav1 - wav2
    
    # get mean and std
    # print(diff_waveform.shape)
    diff_waveform = diff_waveform.detach().cpu().numpy()
    mean = np.mean(diff_waveform)
    std = np.std(diff_waveform)
    noise_wav = np.random.normal(loc=mean, scale=std, size=diff_waveform.shape)
    noise_wav = torch.from_numpy(noise_wav).float()
    
    noise_wav += wav2

    # print(noise_wav.shape)
    # Save the difference waveform as a new .wav file
    output_filename = sys.argv[3]
    torchaudio.save(output_filename, noise_wav, 16000)

if __name__ == '__main__':
    main()
