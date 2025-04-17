from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
import subprocess
import os
import json
import yaml
import uuid
import asyncio
from datetime import datetime
from config import (
    exp_root,
    infer_device,
    is_half,
    is_share,
    python_exec,
    webui_port_infer_tts,
    webui_port_main,
    webui_port_subfix,
    webui_port_uvr5,
)
from tools.asr.config import asr_dict  # 添加 ASR 配置导入
import torch
import torchaudio
import glob

app = FastAPI(title="GPT-SoVITS Training API")

# 任务状态存储
task_status: Dict[str, Dict] = {}

# 数据模型定义
class AudioSliceRequest(BaseModel):
    input_path: str
    output_root: str
    threshold: str = "-34"
    min_length: str = "4000"
    min_interval: str = "300"
    hop_size: str = "10"
    max_sil_kept: str = "500"
    _max: float = 0.9
    alpha: float = 0.25
    n_process: int = 4

class DenoiseRequest(BaseModel):
    input_dir: str
    output_dir: str

class ASRRequest(BaseModel):
    input_dir: str
    output_dir: str
    model: str = "达摩 ASR (中文)"
    model_size: str = "large"
    language: str = "zh"
    precision: str = "float32"

class LabelRequest(BaseModel):
    path_list: str

class TrainingRequest(BaseModel):
    exp_name: str
    input_text: str
    input_wav_dir: str
    gpu_numbers: str
    bert_pretrained_dir: str
    ssl_pretrained_dir: str
    pretrained_s2G_path: str
    i_part: str = "0"  # 添加默认值
    all_parts: str = "1"  # 添加默认值

class SoVITSTrainingRequest(BaseModel):
    exp_name: str
    batch_size: int
    total_epoch: int
    version: str = "v2"  # 新增版本参数
    text_low_lr_rate: Optional[float] = 0.4  # 可选参数，仅v1/v2使用
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    save_every_epoch: int
    gpu_numbers: str
    pretrained_s2G: str
    pretrained_s2D: str
    if_grad_ckpt: Optional[bool] = False  # 可选参数，仅v3使用
    lora_rank: Optional[str] = "32"  # 可选参数，仅v3使用

class GPTTrainingRequest(BaseModel):
    exp_name: str
    batch_size: int
    total_epoch: int
    if_dpo: bool = False
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    save_every_epoch: int
    gpu_numbers: str
    pretrained_s1: str

class TaskStatus(BaseModel):
    status: str  # "running", "completed", "failed"
    message: str
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None

async def filter_audio_length(wav_dir: str, min_length: float = 3.0, max_length: float = 8.0):
    """过滤音频文件，删除长度不在指定范围内的文件"""
    print(f"开始过滤音频文件，保留长度在{min_length}秒到{max_length}秒之间的文件...")
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    removed_count = 0
    kept_count = 0
    
    for wav_file in wav_files:
        try:
            # 加载音频文件
            waveform, sample_rate = torchaudio.load(wav_file)
            duration = waveform.shape[1] / sample_rate
            
            # 检查音频长度
            if duration < min_length or duration > max_length:
                os.remove(wav_file)
                removed_count += 1
            else:
                kept_count += 1
                
        except Exception as e:
            print(f"处理文件 {wav_file} 时出错: {str(e)}")
            continue
    
    print(f"音频过滤完成: 删除了 {removed_count} 个文件，保留了 {kept_count} 个文件")
    return kept_count > 0  # 返回是否还有剩余文件

# API路由
@app.post("/audio/slice")
async def slice_audio(request: AudioSliceRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "running",
        "message": "音频切分中...",
        "start_time": datetime.now(),
        "end_time": None,
        "error": None
    }
    
    async def run_slice():
        try:
            # 1. 音频切分
            cmd = f'"{python_exec}" tools/slice_audio.py "{request.input_path}" "{request.output_root}" {request.threshold} {request.min_length} {request.min_interval} {request.hop_size} {request.max_sil_kept} {request._max} {request.alpha} 0 {request.n_process}'
            process = await asyncio.create_subprocess_shell(cmd)
            await process.wait()

            # 2. 过滤音频长度
            has_valid_files = await filter_audio_length(request.output_root)
            if not has_valid_files:
                raise Exception("没有符合长度要求的音频文件")

            task_status[task_id].update({
                "status": "completed",
                "message": "音频切分和过滤完成",
                "end_time": datetime.now()
            })
        except Exception as e:
            task_status[task_id].update({
                "status": "failed",
                "message": "音频处理失败",
                "end_time": datetime.now(),
                "error": str(e)
            })
    
    background_tasks.add_task(run_slice)
    return {"task_id": task_id}

@app.post("/audio/denoise")
async def denoise_audio(request: DenoiseRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "running",
        "message": "音频降噪中...",
        "start_time": datetime.now(),
        "end_time": None,
        "error": None
    }
    
    async def run_denoise():
        try:
            cmd = f'"{python_exec}" tools/cmd-denoise.py -i "{request.input_dir}" -o "{request.output_dir}" -p {"float16" if is_half else "float32"}'
            process = await asyncio.create_subprocess_shell(cmd)
            await process.wait()
            task_status[task_id].update({
                "status": "completed",
                "message": "音频降噪完成",
                "end_time": datetime.now()
            })
        except Exception as e:
            task_status[task_id].update({
                "status": "failed",
                "message": "音频降噪失败",
                "end_time": datetime.now(),
                "error": str(e)
            })
    
    background_tasks.add_task(run_denoise)
    return {"task_id": task_id}

@app.post("/asr")
async def asr_process(request: ASRRequest):
    try:
        # 获取 ASR 模型配置
        if request.model not in asr_dict:
            raise HTTPException(status_code=400, detail=f"不支持的 ASR 模型: {request.model}")
            
        model_config = asr_dict[request.model]
        
        # 检查模型大小是否支持
        if request.model_size not in model_config["size"]:
            raise HTTPException(status_code=400, detail=f"不支持的模型大小: {request.model_size}")
            
        # 检查语言是否支持
        if request.language not in model_config["lang"]:
            raise HTTPException(status_code=400, detail=f"不支持的语言: {request.language}")
            
        # 检查精度是否支持
        if request.precision not in model_config["precision"]:
            raise HTTPException(status_code=400, detail=f"不支持的精度: {request.precision}")
        
        # 构建命令
        cmd = f'"{python_exec}" tools/asr/{model_config["path"]}'
        cmd += f' -i "{request.input_dir}"'
        cmd += f' -o "{request.output_dir}"'
        cmd += f' -s {request.model_size}'
        cmd += f' -l {request.language}'
        cmd += f' -p {request.precision}'
        
        # 执行命令
        process = await asyncio.create_subprocess_shell(cmd)
        await process.wait()
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail="ASR 处理失败")
            
        return {
            "status": "success",
            "message": "ASR 处理完成",
            "result": {
                "input_dir": request.input_dir,
                "output_dir": request.output_dir,
                "model": request.model,
                "model_size": request.model_size,
                "language": request.language,
                "precision": request.precision
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR 处理失败: {str(e)}")

@app.post("/training/prepare")
async def prepare_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "running",
        "message": "准备训练数据中...",
        "start_time": datetime.now(),
        "end_time": None,
        "error": None
    }
    
    async def run_prepare():
        try:
            # 1a: 文本分词与特征提取
            cmd_1a = f'"{python_exec}" GPT_SoVITS/prepare_datasets/1-get-text.py'
            env_1a = {
                "inp_text": request.input_text,
                "inp_wav_dir": request.input_wav_dir,
                "exp_name": request.exp_name,
                "opt_dir": f"{exp_root}/{request.exp_name}",
                "bert_pretrained_dir": request.bert_pretrained_dir,
                "is_half": str(is_half),
                "_CUDA_VISIBLE_DEVICES": request.gpu_numbers,
                "i_part": request.i_part,
                "all_parts": request.all_parts
            }
            process_1a = await asyncio.create_subprocess_shell(cmd_1a, env={**os.environ, **env_1a})
            await process_1a.wait()

            # 1b: 语音自监督特征提取
            cmd_1b = f'"{python_exec}" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
            env_1b = {
                "inp_text": request.input_text,
                "inp_wav_dir": request.input_wav_dir,
                "exp_name": request.exp_name,
                "opt_dir": f"{exp_root}/{request.exp_name}",
                "cnhubert_base_dir": request.ssl_pretrained_dir,
                "is_half": str(is_half),
                "_CUDA_VISIBLE_DEVICES": request.gpu_numbers,
                "i_part": request.i_part,
                "all_parts": request.all_parts
            }
            process_1b = await asyncio.create_subprocess_shell(cmd_1b, env={**os.environ, **env_1b})
            await process_1b.wait()

            # 1c: 语义Token提取
            cmd_1c = f'"{python_exec}" GPT_SoVITS/prepare_datasets/3-get-semantic.py'
            env_1c = {
                "inp_text": request.input_text,
                "exp_name": request.exp_name,
                "opt_dir": f"{exp_root}/{request.exp_name}",
                "pretrained_s2G": request.pretrained_s2G_path,
                "s2config_path": "GPT_SoVITS/configs/s2.json",
                "is_half": str(is_half),
                "_CUDA_VISIBLE_DEVICES": request.gpu_numbers,
                "i_part": request.i_part,
                "all_parts": request.all_parts
            }
            process_1c = await asyncio.create_subprocess_shell(cmd_1c, env={**os.environ, **env_1c})
            await process_1c.wait()

            task_status[task_id].update({
                "status": "completed",
                "message": "训练集准备完成",
                "end_time": datetime.now()
            })
        except Exception as e:
            task_status[task_id].update({
                "status": "failed",
                "message": "训练集准备失败",
                "end_time": datetime.now(),
                "error": str(e)
            })
    
    background_tasks.add_task(run_prepare)
    return {"task_id": task_id}

@app.post("/training/sovits")
async def train_sovits(request: SoVITSTrainingRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "running",
        "message": "SoVITS训练中...",
        "start_time": datetime.now(),
        "end_time": None,
        "error": None
    }
    
    async def run_sovits():
        try:
            with open("GPT_SoVITS/configs/s2.json") as f:
                data = json.load(f)
            
            s2_dir = f"{exp_root}/{request.exp_name}"
            os.makedirs(f"{s2_dir}/logs_s2", exist_ok=True)
            
            if not is_half:
                data["train"]["fp16_run"] = False
                request.batch_size = max(1, request.batch_size // 2)
                
            data["train"].update({
                "batch_size": request.batch_size,
                "epochs": request.total_epoch,
                "if_save_latest": request.if_save_latest,
                "if_save_every_weights": request.if_save_every_weights,
                "save_every_epoch": request.save_every_epoch,
                "gpu_numbers": request.gpu_numbers,
                "pretrained_s2G": request.pretrained_s2G,
                "pretrained_s2D": request.pretrained_s2D
            })

            # 根据版本设置不同参数
            if request.version in ["v1", "v2"]:
                data["train"]["text_low_lr_rate"] = request.text_low_lr_rate
                data["model"]["version"] = request.version
                data["save_weight_dir"] = "SoVITS_weights_v2" if request.version == "v2" else "SoVITS_weights"
                cmd = f'"{python_exec}" GPT_SoVITS/s2_train.py'
            else:  # v3
                data["train"]["grad_ckpt"] = request.if_grad_ckpt
                data["train"]["lora_rank"] = request.lora_rank
                data["model"]["version"] = "v3"
                data["save_weight_dir"] = "SoVITS_weights_v3"
                cmd = f'"{python_exec}" GPT_SoVITS/s2_train_v3_lora.py'
            
            data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
            data["name"] = request.exp_name
            
            # 创建临时配置文件
            tmp_config_path = f"{s2_dir}/tmp_s2.json"
            with open(tmp_config_path, "w") as f:
                json.dump(data, f)
                
            # 添加配置文件参数到命令
            cmd += f' --config "{tmp_config_path}"'
            
            process = await asyncio.create_subprocess_shell(cmd)
            await process.wait()
            
            task_status[task_id].update({
                "status": "completed",
                "message": "SoVITS训练完成",
                "end_time": datetime.now()
            })
        except Exception as e:
            task_status[task_id].update({
                "status": "failed",
                "message": "SoVITS训练失败",
                "end_time": datetime.now(),
                "error": str(e)
            })
    
    background_tasks.add_task(run_sovits)
    return {"task_id": task_id}

@app.post("/training/gpt")
async def train_gpt(request: GPTTrainingRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "running",
        "message": "GPT训练中...",
        "start_time": datetime.now(),
        "end_time": None,
        "error": None
    }
    
    async def run_gpt():
        try:
            config_path = "GPT_SoVITS/configs/s1longer-v2.yaml"
            with open(config_path) as f:
                data = yaml.safe_load(f)
                
            s1_dir = f"{exp_root}/{request.exp_name}"
            os.makedirs(f"{s1_dir}/logs_s1", exist_ok=True)
            
            if not is_half:
                data["train"]["precision"] = "32"
                request.batch_size = max(1, request.batch_size // 2)
                
            data["train"].update({
                "batch_size": request.batch_size,
                "epochs": request.total_epoch,
                "save_every_n_epoch": request.save_every_epoch,
                "if_save_every_weights": request.if_save_every_weights,
                "if_save_latest": request.if_save_latest,
                "if_dpo": request.if_dpo
            })
            
            data["pretrained_s1"] = request.pretrained_s1
            data["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
            data["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
            data["output_dir"] = f"{s1_dir}/logs_s1"
            
            os.environ["_CUDA_VISIBLE_DEVICES"] = request.gpu_numbers
            os.environ["hz"] = "25hz"
            
            tmp_config_path = "tmp_s1.yaml"
            with open(tmp_config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
                
            cmd = f'"{python_exec}" GPT_SoVITS/s1_train.py --config_file "{tmp_config_path}"'
            process = await asyncio.create_subprocess_shell(cmd)
            await process.wait()
            
            task_status[task_id].update({
                "status": "completed",
                "message": "GPT训练完成",
                "end_time": datetime.now()
            })
        except Exception as e:
            task_status[task_id].update({
                "status": "failed",
                "message": "GPT训练失败",
                "end_time": datetime.now(),
                "error": str(e)
            })
    
    background_tasks.add_task(run_gpt)
    return {"task_id": task_id}

# 添加状态查询端点
@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task_status[task_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 