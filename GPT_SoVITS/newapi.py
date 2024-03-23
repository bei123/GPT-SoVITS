import argparse
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import signal
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import config as global_config
import re
from fastapi import File, UploadFile
import shutil
from flask import Flask, jsonify
import random

g_config = global_config.Config()

# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")

parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")

parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu / mps")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False

parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()

sovits_path = args.sovits_path
gpt_path = args.gpt_path


class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = args.default_refer_path
        self.text = args.default_refer_text
        self.language = args.default_refer_language

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)


default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

device = args.device
port = args.port
host = args.bind_addr

if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    print(f"[WARN] 未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    print(f"[WARN] 未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    print("[INFO] 未指定默认参考音频")
else:
    print(f"[INFO] 默认参考音频路径: {default_refer.path}")
    print(f"[INFO] 默认参考音频文本: {default_refer.text}")
    print(f"[INFO] 默认参考音频语种: {default_refer.language}")

is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback

print(f"[INFO] 半精: {is_half}")

cnhubert_base_path = args.hubert_path
bert_path = args.bert_path

cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True

def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)
def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)





n_semantic = 1024
dict_s2 = torch.load(sovits_path, map_location="cpu")
hps = dict_s2["config"]


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(gpt_path, map_location="cpu")
config = dict_s1["config"]
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
if is_half:
    vq_model = vq_model.half().to(device)
else:
    vq_model = vq_model.to(device)
vq_model.eval()
print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
hz = 50
max_sec = config['data']['max_sec']
t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if is_half:
    t2s_model = t2s_model.half()
t2s_model = t2s_model.to(device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))


if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)

from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language = {
    ("中文"): "all_zh",#全部按中文识别
    ("英文"): "en",#全部按英文识别#######不变
    ("日文"): "all_ja",#全部按日文识别
    ("中英混合"): "zh",#按中英混合识别####不变
    ("日英混合"): "ja",#按日英混合识别####不变
    ("多语种混合"): "auto",#多语种启动切分识别语种
}

cut_method = {
    ("不切"):"cut0",
    ("凑四句一切"): "cut1",
    ("凑50字一切"): "cut2",
    ("按中文句号。切"): "cut3",
    ("按英文句号.切"): "cut4",
    ("按标点符号切"): "cut5",
}

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path
    
tts_pipline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path



def inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,
              seed,
              ):
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
    }
    for item in tts_pipline.run(inputs):
        yield item, actual_seed


def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def handle_change(path, text, language):
    if is_empty(path, text, language):
        return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)

    if path != "" or path is not None:
        default_refer.path = path
    if text != "" or text is not None:
        default_refer.text = text
    if language != "" or language is not None:
        default_refer.language = language

    print(f"[INFO] 当前默认参考音频路径: {default_refer.path}")
    print(f"[INFO] 当前默认参考音频文本: {default_refer.text}")
    print(f"[INFO] 当前默认参考音频语种: {default_refer.language}")
    print(f"[INFO] is_ready: {default_refer.is_ready()}")

    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def handle(text, text_lang, 
              refer_wav_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,seed):
    
   # print("handle=============",text, text_lang)
   # print("音频",refer_wav_path)
    # 检查是否有必要的参数缺失，并分别返回具体缺失的参数信息
    if refer_wav_path == "" or refer_wav_path is None:
        error_message = "缺少必要的参考音频路径。"
    elif prompt_text == "" or prompt_text is None:
        error_message = "缺少必要的提示文本。"
    elif prompt_lang == "" or prompt_lang is None:
        error_message = "缺少必要的提示语言。"
    else:
        error_message = ""

    # 如果有任一必要参数缺失
    if error_message:
        # 尝试使用默认参考音频
        if default_refer.is_ready():
            refer_wav_path, prompt_text, prompt_lang = (
                default_refer.path,
                default_refer.text,
                default_refer.language,
            )
        else:
            # 如果默认参考音频也不可用，则返回具体缺失的参数信息
            return JSONResponse({"code": 400, "message": f"未指定参考音频且接口无预设。{error_message}"}, status_code=400)


    with torch.no_grad():
        
        gen = inference(
            text, text_lang, 
              refer_wav_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,seed
        )

        audio,_ = next(gen)
    sampling_rate,audio_data=audio

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)
    torch.cuda.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")


app = FastAPI()

#clark新增-----2024-02-21
#可在启动后动态修改模型，以此满足同一个api不同的朗读者请求
@app.post("/set_model")
async def set_model(request: Request):
    json_post_raw = await request.json()
    global gpt_path
    gpt_path=json_post_raw.get("gpt_model_path")
    global sovits_path
    sovits_path=json_post_raw.get("sovits_model_path")
    print("gptpath"+gpt_path+";vitspath"+sovits_path)
    change_sovits_weights(sovits_path)
    change_gpt_weights(gpt_path)
    return "ok"
# 新增-----end------

@app.post("/control")
async def control(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))


@app.get("/control")
async def control(command: str = None):
    return handle_control(command)


@app.post("/change_refer")
async def change_refer(request: Request):
    json_post_raw = await request.json()
    return handle_change(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_lang")
    )


@app.get("/change_refer")
async def change_refer(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_lang: str = None
):
    return handle_change(refer_wav_path, prompt_text, prompt_lang)


@app.post("/upload_video")
async def upload_file(file: UploadFile = File(...)):
    
    safe_filename = os.path.basename(file.filename)
    
    upload_path = f"uploads/video/"
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
  
    try:
        with open(os.path.join(upload_path, safe_filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except FileNotFoundError:
        return {"message": "An error occurred while accessing the file."}
    except PermissionError:
        return {"message": "Permission denied while uploading the file."}
    except Exception as e:
        return {"message": f"An error occurred while uploading the file. {str(e)}"}
    
    return {"message": f"Successfully uploaded {safe_filename}."}

@app.post("/upload_text")
async def upload_file(file: UploadFile = File(...)):
    
    safe_filename = os.path.basename(file.filename)
    
    upload_path = f"uploads/text/"
    
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
  
    try:
        with open(os.path.join(upload_path, safe_filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except FileNotFoundError:
        return {"message": "An error occurred while accessing the file."}
    except PermissionError:
        return {"message": "Permission denied while uploading the file."}
    except Exception as e:
        return {"message": f"An error occurred while uploading the file. {str(e)}"}
    
    return {"message": f"Successfully uploaded {safe_filename}."}

@app.post('/test_list_files')
def list_files():
    directory = 'uploads/text/'  
    files = []

    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)

    return jsonify(files)



@app.get("/video_list_files")
async def list_files():
    directory = 'uploads/video/'  

    files = [os.path.join(root, filename)
             for root, _, filenames in os.walk(directory)
             for filename in filenames]

    return files

@app.get("/text_list_files")
async def list_files():
    directory = 'uploads/text/'

    files = [os.path.join(root, filename)
             for root, _, filenames in os.walk(directory)
             for filename in filenames]

    return files


@app.get("/GPT_list_files")
async def list_files():
    directory = 'GPT_weights'  

    files = [os.path.join(root, filename)
             for root, _, filenames in os.walk(directory)
             for filename in filenames]

    return files

@app.get("/SOVITS_list_files")
async def list_files():
    directory = 'SoVITS_weights'

    files = [os.path.join(root, filename)
             for root, _, filenames in os.walk(directory)
             for filename in filenames]

    return files


@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()

    return handle(
        json_post_raw.get("text"),
        json_post_raw.get("text_lang"),
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_lang"),
        json_post_raw.get("top_k"),
        json_post_raw.get("top_p"),
        json_post_raw.get("temperature"),
        json_post_raw.get("text_split_method"),
        json_post_raw.get("batch_size"),
        json_post_raw.get("speed_factor"),
        json_post_raw.get("ref_text_free"),
        json_post_raw.get("split_bucket"),
        json_post_raw.get("fragment_interval"),
        json_post_raw.get("seed")
    )


@app.get("/")
async def tts_endpoint(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_lang: str = None,
        text: str = None,
        text_lang: str = None,
        top_k: int = None,
        top_p: float = None,
        temperature: float = None,
        text_split_method: str = None,
        batch_size: int = None,
        speed_factor: float = None,
        ref_text_free: bool = None,
        split_bucket: bool = None,
        fragment_interval: float = None,
        seed: int = None,
):
    
    return handle(text, text_lang, 
              refer_wav_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,seed)


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=1)