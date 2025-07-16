"""
# api.py usage

` python api.py -dr "123.wav" -dt "一二三。" -dl "zh" `

## 执行参数:

`-s` - `SoVITS模型路径, 可在 config.py 中指定`
`-g` - `GPT模型路径, 可在 config.py 中指定`

调用请求缺少参考音频时使用
`-dr` - `默认参考音频路径`
`-dt` - `默认参考音频文本`
`-dl` - `默认参考音频语种, "中文","英文","日文","韩文","粤语,"zh","en","ja","ko","yue"`

`-d` - `推理设备, "cuda","cpu"`
`-a` - `绑定地址, 默认"127.0.0.1"`
`-p` - `绑定端口, 默认9880, 可在 config.py 中指定`
`-fp` - `覆盖 config.py 使用全精度`
`-hp` - `覆盖 config.py 使用半精度`
`-sm` - `流式返回模式, 默认不启用, "close","c", "normal","n", "keepalive","k"`
·-mt` - `返回的音频编码格式, 流式默认ogg, 非流式默认wav, "wav", "ogg", "aac"`
·-st` - `返回的音频数据类型, 默认int16, "int16", "int32"`
·-cp` - `文本切分符号设定, 默认为空, 以",.，。"字符串的方式传入`

`-hb` - `cnhubert路径`
`-b` - `bert路径`

## 调用:

### 推理

endpoint: `/`

使用执行参数指定的参考音频:
GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

使用执行参数指定的参考音频并设定分割符号:
GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh&cut_punc=，。`
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh",
    "cut_punc": "，。",
}
```

手动指定当次推理所使用的参考音频:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh&text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh",
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

手动指定当次推理所使用的参考音频，并提供参数:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh&text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh&top_k=20&top_p=0.6&temperature=0.6&speed=1&inp_refs="456.wav"&inp_refs="789.wav"`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh",
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh",
    "top_k": 20,
    "top_p": 0.6,
    "temperature": 0.6,
    "speed": 1,
    "inp_refs": ["456.wav","789.wav"]
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400


### 更换默认参考音频

endpoint: `/change_refer`

key与推理端一样

GET:
    `http://127.0.0.1:9880/change_refer?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh"
}
```

RESP:
成功: json, http code 200
失败: json, 400


### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
    `http://127.0.0.1:9880/control?command=restart`
POST:
```json
{
    "command": "restart"
}
```

RESP: 无

"""


import argparse
import os,re
import sys
from datetime import datetime
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import signal
from text.LangSegmenter import LangSegmenter
from time import time as ttime
import torch, torchaudio
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn, SynthesizerTrnV3,Generator
from peft import LoraConfig, PeftModel, get_peft_model
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
import config as global_config
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import time
from logging import getLogger
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple
import gc
import threading
import weakref
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# 添加全局变量
model_version = "v2"  # 默认版本

class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = args.default_refer_path
        self.text = args.default_refer_text
        self.language = args.default_refer_language

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)


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


def init_bigvgan():
    global bigvgan_model,hifigan_model
    from BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=True,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if hifigan_model:
        hifigan_model=hifigan_model.cpu()
        hifigan_model=None
        try:torch.cuda.empty_cache()
        except:pass
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


resample_transform_dict={}
def resample(audio_tensor, sr0,sr1):
    global resample_transform_dict
    key="%s-%s"%(sr0,sr1)
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


from module.mel_processing import spectrogram_torch,mel_spectrogram_torch
spec_min = -12
spec_max = 2
def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1
def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min
mel_fn=lambda x: mel_spectrogram_torch(x, **{
    "n_fft": 1024,
    "win_size": 1024,
    "hop_size": 256,
    "num_mels": 100,
    "sampling_rate": 24000,
    "fmin": 0,
    "fmax": None,
    "center": False
})

mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


sr_model=None
def audio_sr(audio,sr):
    global sr_model
    if sr_model==None:
        from tools.audio_sr import AP_BWE
        try:
            sr_model=AP_BWE(device,DictToAttrRecursive)
        except FileNotFoundError:
            logger.info("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载")
            return audio.cpu().detach().numpy(),sr
    return sr_model(audio,sr)


# 缓存配置和线程安全锁
GPU_CACHE_SIZE = 10
MEMORY_CACHE = {}
cache_lock = threading.RLock()

class ModelManager:
    """模型管理器,负责模型的加载、卸载和缓存管理"""
    
    def __init__(self):
        self.gpu_cache = {}
        self.memory_cache = {}
        self.cache_order = []
        self.lock = threading.RLock()
    
    def _move_to_cpu(self, model):
        """将模型移动到CPU"""
        for attr in ['vq_model', 't2s_model']:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                setattr(model, attr, getattr(model, attr).cpu())
        return model
    
    def _release_gpu_memory(self):
        """强制释放GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"显存已释放: {before:.2f}GB -> {after:.2f}GB")
    
    def get_model(self, key, load_func, *args, **kwargs):
        """获取模型,优先从缓存获取,否则加载"""
        with self.lock:
            # 从GPU缓存获取
            if key in self.gpu_cache:
                self.cache_order.remove(key)
                self.cache_order.append(key)
                logger.info(f"从GPU缓存获取模型: {key}")
                return self.gpu_cache[key]
            
            # 从内存缓存获取并加载到GPU
            if key in self.memory_cache:
                model = self.memory_cache.pop(key)
                self._ensure_gpu_cache_space()
                device = kwargs.get('device', 'cuda')
                is_half = kwargs.get('is_half', False)
                model = self._load_to_gpu(model, device, is_half)
                self.gpu_cache[key] = model
                self.cache_order.append(key)
                logger.info(f"从内存缓存加载模型到GPU: {key}")
                return model
            
            # 从磁盘加载
            try:
                model = load_func(*args, **kwargs)
                self._ensure_gpu_cache_space()
                self.gpu_cache[key] = model
                self.cache_order.append(key)
                logger.info(f"从磁盘加载模型: {key}")
                return model
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                self._release_gpu_memory()
                raise
    
    def _ensure_gpu_cache_space(self):
        """确保GPU缓存有足够空间"""
        while len(self.gpu_cache) >= GPU_CACHE_SIZE and self.cache_order:
            oldest_key = self.cache_order.pop(0)
            if oldest_key in self.gpu_cache:
                model = self._move_to_cpu(self.gpu_cache.pop(oldest_key))
                self.memory_cache[oldest_key] = model
                logger.info(f"将模型从GPU移动到内存: {oldest_key}")
                self._release_gpu_memory()
    
    def _load_to_gpu(self, model, device, is_half=False):
        """将模型加载到GPU"""
        for attr in ['vq_model', 't2s_model']:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                model_part = getattr(model, attr)
                model_part = model_part.half().to(device) if is_half else model_part.to(device)
                setattr(model, attr, model_part)
        return model
    
    def clear(self):
        """清空所有缓存"""
        with self.lock:
            for cache in [self.gpu_cache, self.memory_cache]:
                for key in list(cache.keys()):
                    model = cache.pop(key)
                    del model
            self.cache_order.clear()
            gc.collect()
            self._release_gpu_memory()
            logger.info("所有模型缓存已清空")

# 创建全局模型管理器实例
model_manager = ModelManager()

class Speaker:
    """说话人配置类"""
    def __init__(self, name, gpt=None, sovits=None, phones=None, bert=None, prompt=None):
        self.name = name
        self.sovits = sovits
        self.gpt = gpt
        self.phones = phones
        self.bert = bert
        self.prompt = prompt
    
    def is_loaded(self) -> bool:
        return self.gpt is not None and self.sovits is not None

class Sovits:
    """SoVITS模型封装类"""
    def __init__(self, vq_model, hps):
        self.vq_model = vq_model
        self.hps = hps

class Gpt:
    """GPT模型封装类"""
    def __init__(self, max_sec, t2s_model):
        self.max_sec = max_sec
        self.t2s_model = t2s_model

# 全局变量
speaker_list: Dict[str, Speaker] = {}
hz = 50

from process_ckpt import get_sovits_version_from_path_fast,load_sovits_new
def load_sovits_model(model_name, device, is_half=False):
    try:
        sovits_path = os.path.join("api_Model", model_name, f"{model_name}.pth")
        if not os.path.exists(sovits_path):
            raise FileNotFoundError(f"SoVITS模型文件不存在: {sovits_path}")

        # 先定义模型版本相关变量
        path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
        path_sovits_v4 = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
        is_exist_s2gv3 = os.path.exists(path_sovits_v3)
        is_exist_s2gv4 = os.path.exists(path_sovits_v4)

        version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
        model_version = model_version.lower()  # 统一小写
        if if_lora_v3 and not is_exist_s2gv3:
            raise FileNotFoundError("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")
        if model_version == "v4" and not is_exist_s2gv4:
            raise FileNotFoundError("SoVITS V4 底模缺失，无法加载相应模型")

        dict_s2 = load_sovits_new(sovits_path)
        if "config" not in dict_s2:
            raise RuntimeError("权重文件缺少 config 字段")
        hps = DictToAttrRecursive(dict_s2["config"])
        hps.model.semantic_frame_rate = "25hz"
        model_params_dict = vars(hps.model)
        if model_version in ["v3", "v4"]:
            vq_model = SynthesizerTrnV3(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **model_params_dict
            )
        else:
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **model_params_dict
            )

        # 初始化声码器
        if model_version == "v3":
            init_bigvgan()
        elif model_version == "v4":
            init_hifigan()

        logger.info(f"模型版本: {hps.model.version}")

        # 清理不需要的模块
        if "pretrained" not in sovits_path and hasattr(vq_model, 'enc_q'):
            delattr(vq_model, 'enc_q')

        # 设备转移
        vq_model = vq_model.half().to(device) if is_half else vq_model.to(device)
        vq_model.eval()

        # LoRA 处理
        if not if_lora_v3:
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
        else:
            path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False)
            lora_rank = dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
            vq_model.cfm = vq_model.cfm.merge_and_unload()
            vq_model.eval()

        return Sovits(vq_model, hps)
    
    except Exception as e:
        logger.error(f"加载SoVITS模型失败: {str(e)}")
        raise

def load_gpt_model(model_name, device, is_half=False):
    """加载GPT模型"""
    try:
        gpt_path = os.path.join("api_Model", model_name, f"{model_name}.ckpt")
        if not os.path.exists(gpt_path):
            raise FileNotFoundError(f"GPT模型文件不存在: {gpt_path}")
        
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.half() if is_half else t2s_model
        t2s_model = t2s_model.to(device)
        t2s_model.eval()

        return Gpt(config["data"]["max_sec"], t2s_model)
    
    except Exception as e:
        logger.error(f"加载GPT模型失败: {str(e)}")
        raise

def get_sovits_weights(model_name: str, device: str, is_half: bool = False) -> Sovits:
    """获取SoVITS模型权重"""
    start_time = time.time()
    cache_key = f"sovits_{model_name}_{device}_{is_half}"
    
    try:
        model = model_manager.get_model(
            cache_key, 
            load_sovits_model, 
            model_name, 
            device=device, 
            is_half=is_half
        )
        
        logger.info(f"get_sovits_weights 耗时: {time.time() - start_time:.2f} 秒")
        return model
    
    except Exception as e:
        logger.error(f"加载SoVITS模型失败: {str(e)}, 耗时: {time.time() - start_time:.2f} 秒")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(f"加载SoVITS模型失败: {str(e)}")

def get_gpt_weights(model_name: str, device: str, is_half: bool = False) -> Gpt:
    """获取GPT模型权重"""
    start_time = time.time()
    cache_key = f"gpt_{model_name}_{device}_{is_half}"
    
    try:
        model = model_manager.get_model(
            cache_key, 
            load_gpt_model, 
            model_name, 
            device=device, 
            is_half=is_half
        )
        
        logger.info(f"get_gpt_weights 耗时: {time.time() - start_time:.2f} 秒")
        return model
    
    except Exception as e:
        logger.error(f"加载GPT模型失败: {str(e)}, 耗时: {time.time() - start_time:.2f} 秒")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(f"加载GPT模型失败: {str(e)}")

def change_gpt_sovits_weights(model_name: str, device: str, is_half: bool = False) -> JSONResponse:
    """切换GPT和SoVITS模型权重"""
    start_time = time.time()
    
    try:
        gpt = get_gpt_weights(model_name, device, is_half)
        sovits = get_sovits_weights(model_name, device, is_half)
        
        speaker_list["default"] = Speaker(name="default", gpt=gpt, sovits=sovits)
        
        logger.info(f"change_gpt_sovits_weights 耗时: {time.time() - start_time:.2f} 秒")
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    
    except Exception as e:
        logger.error(f"切换模型失败: {str(e)}, 耗时: {time.time() - start_time:.2f} 秒")
        model_manager._release_gpu_memory()
        return JSONResponse({"code": 400, "message": str(e)}, status_code=400)

def cleanup_resources():
    """清理所有模型资源"""
    logger.info("开始清理所有模型资源...")
    model_manager.clear()
    speaker_list.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("所有模型资源已清理完毕")

def initialize_app():
    """应用初始化"""
    logger.info("初始化应用...")
    
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
        
        device_count = torch.cuda.device_count()
        logger.info(f"检测到 {device_count} 个GPU设备")
        for i in range(device_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        torch.zeros(1).cuda()
        torch.cuda.empty_cache()
    
    logger.info("应用初始化完成")

def periodic_cleanup():
    """定期清理"""
    logger.info("执行定期清理...")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"当前GPU内存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        if len(model_manager.gpu_cache) > GPU_CACHE_SIZE // 2:
            keep_count = max(1, len(model_manager.gpu_cache) // 2)
            to_remove = model_manager.cache_order[:-keep_count]
            
            for key in to_remove:
                if key in model_manager.gpu_cache:
                    model = model_manager._move_to_cpu(model_manager.gpu_cache.pop(key))
                    model_manager.memory_cache[key] = model
                    model_manager.cache_order.remove(key)
                    logger.info(f"定期清理: 将模型 {key} 从GPU移动到内存")
            
            model_manager._release_gpu_memory()
    
    logger.info("定期清理完成")

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


def clean_text_inf(text, language, version):
    language = language.replace("all_","")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert

from text import chinese
def get_phones_and_bert(text,language,version,final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(torch.float16 if is_half == True else torch.float32),norm_text


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def init_hifigan():
    global hifigan_model,bigvgan_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0, is_bias=True
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load("%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,), map_location="cpu")
    print("loading vocoder",hifigan_model.load_state_dict(state_dict_g))
    if bigvgan_model:
        bigvgan_model=bigvgan_model.cpu()
        bigvgan_model=None
        try:torch.cuda.empty_cache()
        except:pass
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)

bigvgan_model=hifigan_model=None
if model_version=="v3":
    init_bigvgan()
if model_version=="v4":
    init_hifigan()

    
# ===== 1. 顶部引入SV模型 =====
from GPT_SoVITS.sv import SV

# ===== 2. 全局变量 =====
sv_cn_model = None

def init_sv_cn():
    global sv_cn_model
    if sv_cn_model is None:
        sv_cn_model = SV(device, is_half)

def get_spepc(hps, filename, is_v2pro=False):
    audio, sr0 = librosa.load(filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    audio_norm = audio.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    if is_v2pro:
        # v2Pro/v2ProPlus需要16k音频做sv_emb
        import torchaudio
        audio_16k, sr1 = torchaudio.load(filename)
        if sr1 != 16000:
            audio_16k = torchaudio.transforms.Resample(sr1, 16000)(audio_16k)
        return spec, audio_16k
    return spec


def pack_audio(audio_bytes, data, rate):
    """
    根据媒体类型打包音频数据，并打印时间
    """
    start_time = time.time()  # 记录开始时间

    if media_type == "ogg":
        audio_bytes = pack_ogg(audio_bytes, data, rate)
    elif media_type == "aac":
        audio_bytes = pack_aac(audio_bytes, data, rate)
    else:
        # wav无法流式, 先暂存raw
        audio_bytes = pack_raw(audio_bytes, data, rate)
    
    elapsed_time = time.time() - start_time  # 计算耗时
    print(f"pack_audio 耗时: {elapsed_time:.2f} 秒")  # 打印耗时

    return audio_bytes


def pack_ogg(audio_bytes, data, rate):
    # Author: AkagawaTsurunaki
    # Issue:
    #   Stack overflow probabilistically occurs
    #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
    #   using the Python library `soundfile`
    # Note:
    #   This is an issue related to `libsndfile`, not this project itself.
    #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
    #   and try to convert it to an ogg file.
    # Related:
    #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
    #   https://github.com/libsndfile/libsndfile/issues/1023
    #   https://github.com/bastibe/python-soundfile/issues/396
    # Suggestion:
    #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

    def handle_pack_ogg():
        with sf.SoundFile(audio_bytes, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
            audio_file.write(data)

    import threading
    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")

    return audio_bytes


def pack_raw(audio_bytes, data, rate):
    audio_bytes.write(data.tobytes())

    return audio_bytes


def pack_wav(audio_bytes, rate):
    if is_int32:
        data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int32)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV', subtype='PCM_32')
    else:
        data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int16)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV')
    return wav_bytes


def pack_aac(audio_bytes, data, rate):
    if is_int32:
        pcm = 's32le'
        bit_rate = '256k'
    else:
        pcm = 's16le'
        bit_rate = '128k'
    process = subprocess.Popen([
        'ffmpeg',
        '-f', pcm,  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', bit_rate,  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    audio_bytes.write(out)

    return audio_bytes


def read_clean_buffer(audio_bytes):
    audio_chunk = audio_bytes.getvalue()
    audio_bytes.truncate(0)
    audio_bytes.seek(0)

    return audio_bytes, audio_chunk


def cut_text(text, punc):
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items)%2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    return text


def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, top_k=15, top_p=0.6, temperature=0.6, speed=1, inp_refs=None, sample_steps=8, if_sr=False, spk="default"):
    infer_sovits = speaker_list[spk].sovits
    vq_model = infer_sovits.vq_model
    hps = infer_sovits.hps
    model_version = hps.model.version
    model_version = model_version.lower()  # 统一小写

    infer_gpt = speaker_list[spk].gpt
    t2s_model = infer_gpt.t2s_model
    max_sec = infer_gpt.max_sec

    # 预处理文本
    prompt_text = prompt_text.strip("\n")
    if prompt_text[-1] not in splits: 
        prompt_text += "。" if prompt_language != "en" else "."
    text = text.strip("\n")
    
    # 设置数据类型
    dtype = torch.float16 if is_half else torch.float32
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half else np.float32)
    
    # 处理参考音频
    with torch.no_grad():
        # 加载并预处理参考音频
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        
        # 根据精度设置转换到设备
        if is_half:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
            # 确保模型也是半精度
            vq_model = vq_model.half()
            t2s_model = t2s_model.half()
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
            # 确保模型是单精度
            vq_model = vq_model.float()
            t2s_model = t2s_model.float()
            
        wav16k = torch.cat([wav16k, zero_wav_torch])
        
        # 提取语义特征
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        # 确保ssl_content的数据类型与模型匹配
        ssl_content = ssl_content.to(dtype)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

        is_v2pro = model_version in ["v2pro", "v2proplus"]
        if model_version not in ["v3", "v4"]:
            refers = []
            sv_emb = [] if is_v2pro else None
            if is_v2pro:
                init_sv_cn()
            if inp_refs:
                for path in inp_refs:
                    try:
                        if is_v2pro:
                            refer, audio_tensor = get_spepc(hps, path, is_v2pro=True)
                            refers.append(refer.to(dtype).to(device))
                            sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor.to(device)))
                        else:
                            refer = get_spepc(hps, path)
                            refers.append(refer.to(dtype).to(device))
                    except Exception as e:
                        logger.error(e)
            if len(refers) == 0:
                if is_v2pro:
                    refer, audio_tensor = get_spepc(hps, ref_wav_path, is_v2pro=True)
                    refers = [refer.to(dtype).to(device)]
                    sv_emb = [sv_cn_model.compute_embedding3(audio_tensor.to(device))]
                else:
                    refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
        else:
            refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)

    # 语言处理
    prompt_language = dict_language[prompt_language.lower()]
    text_language = dict_language[text_language.lower()]
    
    # 获取音素和BERT特征
    phones1, bert1, _ = get_phones_and_bert(prompt_text, prompt_language, model_version)
    
    # 分割文本处理
    texts = text.split("\n")
    all_audio = []
    
    for text in texts:
        # 跳过纯符号文本
        if only_punc(text):
            continue

        # 处理文本
        if text[-1] not in splits: 
            text += "。" if text_language != "en" else "."
            
        phones2, bert2, _ = get_phones_and_bert(text, text_language, model_version)
        bert = torch.cat([bert1, bert2], 1)
        
        # 确保bert特征的数据类型与模型匹配
        bert = bert.to(dtype)

        # 准备模型输入
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        
        # 生成语义特征
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        # 根据版本解码音频
        if model_version not in ["v3", "v4"]:
            if is_v2pro:
                audio = vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(device).unsqueeze(0),
                    refers,
                    speed=speed,
                    sv_emb=sv_emb,
                ).detach().cpu().numpy()[0, 0]
            else:
                audio = vq_model.decode(
                    pred_semantic, 
                    torch.LongTensor(phones2).to(device).unsqueeze(0),
                    refers,
                    speed=speed
                ).detach().cpu().numpy()[0, 0]
        else:
            refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)
            phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio = ref_audio.to(device).float()
            if ref_audio.shape[0] == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            tgt_sr=24000 if model_version=="v3"else 32000
            if sr != tgt_sr:
                ref_audio = resample(ref_audio, sr,tgt_sr)
            mel2 = mel_fn(ref_audio)if model_version=="v3"else mel_fn_v4(ref_audio)
            mel2 = norm_spec(mel2)
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            Tref=468 if model_version=="v3"else 500
            Tchunk=934 if model_version=="v3"else 1000
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            chunk_len = Tchunk - T_min
            mel2 = mel2.to(dtype)
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
            
            # 分块处理长音频
            cfm_resss = []
            idx = 0
            while True:
                fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                if fea_todo_chunk.shape[-1] == 0: 
                    break
                    
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                
                cfm_res = vq_model.cfm.inference(
                    fea, 
                    torch.LongTensor([fea.size(1)]).to(fea.device), 
                    mel2, 
                    sample_steps, 
                    inference_cfg_rate=0
                )
                
                cfm_res = cfm_res[:, :, mel2.shape[2]:]
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
                
            cfm_res = torch.cat(cfm_resss, 2)
            cfm_res = denorm_spec(cfm_res)
            
            # 初始化声码器
            if model_version == "v3":
                if bigvgan_model is None:
                    init_bigvgan()
                vocoder_model = bigvgan_model
            else:  # v4
                if hifigan_model is None:
                    init_hifigan()
                vocoder_model = hifigan_model
                
            with torch.inference_mode():
                wav_gen = vocoder_model(cfm_res)
                audio = wav_gen[0][0].cpu().detach().numpy()

        # 归一化音频
        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio
            
        # 添加静音
        all_audio.append(audio)
        all_audio.append(zero_wav)

    # 合并所有音频
    audio_opt = np.concatenate(all_audio, 0)

    # 采样率处理
    if model_version in {"v1","v2"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    elif model_version in ["v2pro", "v2proplus"]:
        opt_sr = 32000
        if if_sr:
            audio_opt = torch.from_numpy(audio_opt).float().to(device)
            audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
            max_audio = np.abs(audio_opt).max()
            if max_audio > 1:
                audio_opt /= max_audio
            opt_sr = 48000
    else:  # v4
        opt_sr = 48000
    
    if if_sr and opt_sr == 24000:
        audio_opt = torch.from_numpy(audio_opt).float().to(device)
        audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
        max_audio = np.abs(audio_opt).max()
        if max_audio > 1:
            audio_opt /= max_audio
        opt_sr = 48000

    # 打包音频
    audio_bytes = BytesIO()
    if is_int32:
        audio_bytes = pack_audio(audio_bytes, (audio_opt * 2147483647).astype(np.int32), opt_sr)
    else:
        audio_bytes = pack_audio(audio_bytes, (audio_opt * 32768).astype(np.int16), opt_sr)
    
    # 流式模式处理
    if stream_mode == "normal":
        audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
        yield audio_chunk
    
    # 非流式模式处理
    if not stream_mode == "normal": 
        if media_type == "wav":
            opt_sr = 48000 if if_sr and model_version in ["v2pro", "v2proplus"] else opt_sr
            audio_bytes = pack_wav(audio_bytes, opt_sr)
        yield audio_bytes.getvalue()

def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

nowLoadModelName = "paimeng"
def handle_change(path, text, language):
    global nowLoadModelName 
    if is_empty(path, text, language):
        return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)

    if path != "" or path is not None:
        default_refer.path = path
    if text != "" or text is not None:
        default_refer.text = text
    if language != "" or language is not None:
        default_refer.language = language

    logger.info(f"当前默认参考音频路径: {default_refer.path}")
    logger.info(f"当前默认参考音频文本: {default_refer.text}")
    logger.info(f"当前默认参考音频语种: {default_refer.language}")
    logger.info(f"is_ready: {default_refer.is_ready()}")


    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def handle(text, text_language,model_name,):
    global nowLoadModelName
    # 生成 refer_wav_path
    refer_wav_path = f"api_Model/{model_name}/{model_name}.wav"
    
    # 获取 prompt_text
    prompt_text = g_config.modelToPromptText.get(model_name, "")
    
    # 设置 prompt_language
    prompt_language = g_config.modelToPromptLanguage.get(model_name, "")
    
    # 默认不剪切标点符号
    cut_punc = None

    # 判断是否切换模型
    if nowLoadModelName != model_name:
        logger.info(f"====当前加载的模型为: {nowLoadModelName}, 切换为模型: {model_name}====")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 只需调用一次 `change_gpt_sovits_weights` 函数
        change_gpt_sovits_weights(model_name, device, is_half)
        
        nowLoadModelName = model_name

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"生成语音参数====={refer_wav_path}, {prompt_text}, {prompt_language}, {text}, {text_language}, {model_name} at {current_time}")
    
    if (
            refer_wav_path == "" or refer_wav_path is None
            or prompt_text == "" or prompt_text is None
            or prompt_language == "" or prompt_language is None
    ):
        refer_wav_path, prompt_text, prompt_language = (
            default_refer.path,
            default_refer.text,
            default_refer.language,
        )
        if not default_refer.is_ready():
            return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

        if not sample_steps in [4,8,16,32]:
         sample_steps = 32

    if cut_punc == None:
        text = cut_text(text,default_cut_punc)
    else:
        text = cut_text(text,cut_punc)

    return StreamingResponse(get_tts_wav(refer_wav_path,prompt_text, prompt_language, text,text_language,), media_type="audio/"+media_type)




# --------------------------------
# 初始化部分
# --------------------------------
dict_language = {
    "中文": "all_zh",
    "粤语": "all_yue",
    "英文": "en",
    "日文": "all_ja",
    "韩文": "all_ko",
    "中英混合": "zh",
    "粤英混合": "yue",
    "日英混合": "ja",
    "韩英混合": "ko",
    "多语种混合": "auto",    #多语种启动切分识别语种
    "多语种混合(粤语)": "auto_yue",
    "all_zh": "all_zh",
    "all_yue": "all_yue",
    "en": "en",
    "all_ja": "all_ja",
    "all_ko": "all_ko",
    "zh": "zh",
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "auto": "auto",
    "auto_yue": "auto_yue",
}

# 支持的模型版本
v3v4set = {"v3", "v4"}

# logger
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger('uvicorn')

# 获取配置
g_config = global_config.Config()

# 获取参数
parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")
parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")
parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False
parser.add_argument("-sm", "--stream_mode", type=str, default="close", help="流式返回模式, close / normal / keepalive")
parser.add_argument("-mt", "--media_type", type=str, default="wav", help="音频编码格式, wav / ogg / aac")
parser.add_argument("-st", "--sub_type", type=str, default="int16", help="音频数据类型, int16 / int32")
parser.add_argument("-cp", "--cut_punc", type=str, default="", help="文本切分符号设定, 符号范围,.;?!、，。？！；：…")
# 切割常用分句符为 `python ./api.py -cp ".?!。？！"`
parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()
sovits_path = args.sovits_path
gpt_path = args.gpt_path
device = args.device
port = args.port
host = args.bind_addr
cnhubert_base_path = args.hubert_path
bert_path = args.bert_path
default_cut_punc = args.cut_punc

# 应用参数配置
default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

# 模型路径检查
if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    logger.warn(f"未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    logger.warn(f"未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    logger.info("未指定默认参考音频")
else:
    logger.info(f"默认参考音频路径: {default_refer.path}")
    logger.info(f"默认参考音频文本: {default_refer.text}")
    logger.info(f"默认参考音频语种: {default_refer.language}")

# 获取半精度
is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback
logger.info(f"半精: {is_half}")

# 流式返回模式
if args.stream_mode.lower() in ["normal","n"]:
    stream_mode = "normal"
    logger.info("流式返回已开启")
else:
    stream_mode = "close"

# 音频编码格式
if args.media_type.lower() in ["aac","ogg"]:
    media_type = args.media_type.lower()
elif stream_mode == "close":
    media_type = "wav"
else:
    media_type = "ogg"
logger.info(f"编码格式: {media_type}")

# 音频数据类型
if args.sub_type.lower() == 'int32':
    is_int32 = True
    logger.info(f"数据类型: int32")
else:
    is_int32 = False
    logger.info(f"数据类型: int16")

# 初始化模型
cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
ssl_model = cnhubert.get_model()
if is_half:
    bert_model = bert_model.half().to(device)
    ssl_model = ssl_model.half().to(device)
else:
    bert_model = bert_model.to(device)
    ssl_model = ssl_model.to(device)
model_name = "paimeng"  # 替换为实际的模型名称
change_gpt_sovits_weights(model_name=model_name,device=device,is_half=is_half)

# 接口部分
# --------------------------------
app = FastAPI()

# 创建请求锁，用于限制QQ机器人请求
qqbot_lock = threading.Lock()
qqbot_processing = False

# 允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法（GET、POST 等）
    allow_headers=["*"],  # 允许所有请求头
)

# Setup logger
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# Create a rotating file handler
handler = RotatingFileHandler("api.log", maxBytes=1073741824, backupCount=3)

# Define the format for the log messages
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Set the formatter for the handler
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# QQ机器人专用处理函数，带锁机制
async def handle_qqbot(text, text_language, model_name):
    global qqbot_processing
    
    # 尝试获取锁，如果已经在处理请求，则等待
    if not qqbot_lock.acquire(blocking=False):
        logger.info("QQ机器人请求被锁定，等待上一个请求完成")
        # 等待锁释放
        while qqbot_processing:
            await asyncio.sleep(0.5)
        qqbot_lock.acquire()
    
    try:
        qqbot_processing = True
        logger.info(f"开始处理QQ机器人请求: {text[:20]}...")
        
        # 调用原有的处理函数
        result = handle(text, text_language, model_name)
        
        logger.info("QQ机器人请求处理完成")
        return result
    finally:
        qqbot_processing = False
        qqbot_lock.release()

# QQ机器人专用端点
@app.post("/qqbot")
async def qqbot_tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return await handle_qqbot(
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
        json_post_raw.get("model_name"), 
    )

@app.get("/qqbot")
async def qqbot_tts_endpoint(
        text: str = None,
        text_language: str = None,
        model_name: str = None,
):
    return await handle_qqbot(text, text_language, model_name)

# 原有的端点保持不变
@app.post("/set_model")
async def set_model(request: Request):
    json_post_raw = await request.json()
    model_name = json_post_raw.get("model_name")
    return change_gpt_sovits_weights(
        model_name=model_name, 
        device=device,
        is_half=is_half
    )

@app.get("/set_model")
async def set_model(
        model_name: str = None,
        
):
    return change_gpt_sovits_weights(model_name=model_name,device=device,is_half=is_half)


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
        json_post_raw.get("prompt_language")
    )


@app.get("/change_refer")
async def change_refer(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_language: str = None
):
    return handle_change(refer_wav_path, prompt_text, prompt_language)


@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return handle(
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
            json_post_raw.get("model_name"), 
    )


@app.get("/")
async def tts_endpoint(
        text: str = None,
        text_language: str = None,
        model_name: str = None,
):
    return handle(text,model_name,text_language)


if __name__ == "__main__":
    # import cProfile
    # import pstats

   
    # profiler = cProfile.Profile()
    # profiler.enable()

    logger.info(f"API服务器已启动，端口: {port}")
    logger.info(f"QQ机器人专用API端点: /qqbot")
    
    # 启动主API服务器
    uvicorn.run(app, host=host, port=port, workers=1)

    # profiler.disable()

    
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')  
    # stats.dump_stats('api_profile.prof')  


    # stats.print_stats()
