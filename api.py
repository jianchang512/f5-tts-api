
import os,time,sys
from pathlib import Path
ROOT_DIR=Path(__file__).parent.as_posix()

# ffmpeg
if sys.platform == 'win32':
    os.environ['PATH'] = ROOT_DIR + f';{ROOT_DIR}\\ffmpeg;' + os.environ['PATH']
else:
    os.environ['PATH'] = ROOT_DIR + f':{ROOT_DIR}/ffmpeg:' + os.environ['PATH']

SANFANG=True
if Path(f"{ROOT_DIR}/modelscache").exists():
    SANFANG=False
    os.environ['HF_HOME']=Path(f"{ROOT_DIR}/modelscache").as_posix()


import re
import torch
from torch.backends import cudnn
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence

from cached_path import cached_path

import soundfile as sf
import io
import tempfile
import logging
import traceback
from waitress import serve
from importlib.resources import files
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT


TMPDIR=(Path(__file__).parent/'tmp').as_posix()
Path(TMPDIR).mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
CORS(app)

# --------------------- Settings -------------------- #



# Add this near the top of the file, after other imports
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def load_model2(repo_name='F5-TTS',vocoder_name='vocos'):
    mel_spec_type = vocoder_name
    model_cfg =  f"{ROOT_DIR}/configs/F5TTS_Base_train.yaml"

    model_cfg = OmegaConf.load(model_cfg).model.arch
    model_cls = DiT

    ckpt_file = ""
    vocab_file=''
    remove_silence=False
    speed=1.0
    if repo_name=='F5-TTS':
        if vocoder_name == "vocos":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))

        elif vocoder_name == "bigvgan":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base_bigvgan"
            ckpt_step = 1250000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))
    else:
        mel_spec_type='vocos'
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        if ckpt_file == "":
            repo_name = "E2-TTS"
            exp_name = "E2TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    model=load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=mel_spec_type, vocab_file=vocab_file)
    return model



# Dictionary to store loaded models
loaded_models = {}



@app.route('/api', methods=['POST'])
def api():
    logger.info("Accessing generate_audio route")
    ref_text = request.form.get('ref_text')
    gen_text = request.form.get('gen_text')
    remove_silence = int(request.form.get('remove_silence',0))
    
    speed = float(request.form.get('speed',1.0))
    model_choice = 'F5-TTS'
    vocoder_name = request.form.get('vocoder_name','vocos')

    
    if not all([ref_text, gen_text, model_choice]):  # Include audio_filename in the check
        return jsonify({"error": "Missing required parameters"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        logger.error("No audio file selected")
        return jsonify({"error": "No audio file selected"}), 400


    logger.info(f"Processing audio file: {audio_file.filename}")
    audio_name=f'{TMPDIR}/{time.time()}-{audio_file.filename}'
    audio_file.save(audio_name)
    
    try:

        if model_choice not in loaded_models:
            loaded_models[model_choice] = load_model2(repo_name=model_choice)


        model = loaded_models[model_choice]
        if vocoder_name == "vocos":
            
            vocoder = load_vocoder(vocoder_name=vocoder_name,
                is_local=True if not SANFANG else False, 
                local_path='./modelscache/hub/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/' if not SANFANG else None
            )
        elif vocoder_name == "bigvgan":
            vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, local_path="./checkpoints/bigvgan_v2_24khz_100band_256x")

        
        main_voice = {"ref_audio": audio_name, "ref_text": ref_text}
        voices = {"main": main_voice}
        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            print("Voice:", voice)
            print("Ref_audio:", voices[voice]["ref_audio"])
            print("Ref_text:", voices[voice]["ref_text"])

        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, gen_text)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio, ref_text, gen_text, model, vocoder, mel_spec_type=vocoder_name, speed=speed
            )
            generated_audio_segments.append(audio)
        
        # 
        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

       
        wave_path=TMPDIR+f'/out-{time.time()}.wav'
        print(f'{wave_path=}')
        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            if remove_silence==1:
                remove_silence_for_generated_wav(f.name)
            print(f.name)

        return send_file(wave_path, mimetype="audio/wav", as_attachment=True, download_name=audio_file.filename)

    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500        



@app.route('/v1/audio/speech', methods=['POST'])
def audio_speech():
    """
    兼容 OpenAI /v1/audio/speech API 的接口
    """
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()

    # 检查请求中是否包含必要的参数
    if 'input' not in data or 'voice' not in data:
        return jsonify({"error": "请求缺少必要的参数： input, voice"}), 400
    

    gen_text = data.get('input')
    speed = float(data.get('speed',1.0))
    
    # 参考音频
    voice = data.get('voice','')
    
    audio_file,ref_text=voice.split('###')

    if not Path(audio_file).exists() or not Path(f'{ROOT_DIR}/{audio_file}').exists():
        return jsonify({"error": {"message": f"必须填写'参考音频路径###参考音频文本'", "type": e.__class__.__name__, "param": f'speed={speed},voice={voice},input={gen_text}', "code": 400}}), 500

    model_choice='F5-TTS'

    try:
        if model_choice not in loaded_models:
            loaded_models[model_choice] = load_model2(repo_name=model_choice)


        model = loaded_models[model_choice]

        vocoder = load_vocoder(vocoder_name='vocos',
            is_local=True if not SANFANG else False, 
            local_path='./modelscache/hub/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/' if not SANFANG else None
        )

        
        main_voice = {"ref_audio": audio_file, "ref_text": ref_text}
        voices = {"main": main_voice}
        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            print("Voice:", voice)
            print("Ref_audio:", voices[voice]["ref_audio"])
            print("Ref_text:", voices[voice]["ref_text"])

        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, gen_text)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio, ref_text, gen_text, model, vocoder, mel_spec_type='vocos', speed=speed
            )
            generated_audio_segments.append(audio)
        
        # 
        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

       
        wave_path=TMPDIR+f'/openai-{time.time()}.wav'
        print(f'{wave_path=}')
        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            print(f.name)

        return send_file(wave_path, mimetype="audio/x-wav")
    except Exception as e:
        return jsonify({"error": {"message": f"{e}", "type": e.__class__.__name__, "param": f'speed={speed},voice={voice},input={text}', "code": 400}}), 500
     


if __name__ == '__main__':
    try:
        host="127.0.0.1"
        port=5010
        print(f"api接口地址  http://{host}:{port}")
        serve(app,host=host, port=port)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())