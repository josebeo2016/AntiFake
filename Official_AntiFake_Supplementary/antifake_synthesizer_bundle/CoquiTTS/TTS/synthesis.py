from TTS.api import TTS
import os
import logging
import sys

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def voice_clone(tts, speaker_wav, text, output_path):

    tts.tts_to_file(text, speaker_wav=speaker_wav, language='en', file_path=output_path)


if __name__ == '__main__':
    
    text = sys.argv[1]
    speaker_wav = sys.argv[2]
    out_path = sys.argv[3]
   
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
    tts.tts_to_file(text, 
                    speaker_wav=speaker_wav, 
                    language='en', 
                    file_path=out_path)
    
    