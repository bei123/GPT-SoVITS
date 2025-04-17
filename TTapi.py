import requests
import time
import os
from typing import Optional

class TrainingClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.current_task_id: Optional[str] = None

    def wait_for_completion(self, timeout: int = 3600, interval: int = 5) -> bool:
        """等待当前任务完成"""
        if not self.current_task_id:
            raise ValueError("没有正在执行的任务")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/status/{self.current_task_id}")
                if response.status_code == 200:
                    status = response.json()
                    print(f"任务状态: {status['status']} - {status['message']}")
                    
                    if status["status"] == "completed":
                        return True
                    elif status["status"] == "failed":
                        raise Exception(f"任务失败: {status.get('error', '未知错误')}")
            except requests.exceptions.RequestException as e:
                print(f"查询状态时出错: {e}")
            
            time.sleep(interval)
        
        raise TimeoutError("任务超时")

    def slice_audio(self, input_path: str, output_root: str) -> None:
        """音频切分"""
        print("开始音频切分...")
        response = requests.post(f"{self.base_url}/audio/slice", json={
            "input_path": input_path,
            "output_root": output_root,
            "threshold": "-34",
            "min_length": "4000",
            "min_interval": "300",
            "hop_size": "10",
            "max_sil_kept": "500",
            "_max": 0.9,
            "alpha": 0.25,
            "n_process": 4
        })
        if response.status_code != 200:
            raise Exception(f"音频切分请求失败: {response.text}")
        
        self.current_task_id = response.json()["task_id"]
        self.wait_for_completion()
        print("音频切分完成")

    def run_asr(self, input_dir: str, output_dir: str, model: str = "达摩 ASR (中文)", model_size: str = "large", language: str = "zh", precision: str = "float32"):
        """
        运行语音识别
        
        Args:
            input_dir: 输入音频目录
            output_dir: 输出目录
            model: ASR 模型名称，默认为"达摩 ASR (中文)"
            model_size: 模型大小，默认为"large"
            language: 语言，默认为"zh"
            precision: 精度，默认为"float16"
        """
        try:
            response = requests.post(f"{self.base_url}/asr", json={
                "input_dir": input_dir,
                "output_dir": output_dir,
                "model": model,
                "model_size": model_size,
                "language": language,
                "precision": precision
            })
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    return result["result"]
                else:
                    raise Exception(result["message"])
            else:
                raise Exception(f"ASR 请求失败: {response.text}")
        except Exception as e:
            raise Exception(f"ASR 处理失败: {str(e)}")

    def prepare_training(self, exp_name: str, input_text: str, input_wav_dir: str, 
                        gpu_numbers: str, bert_pretrained_dir: str, 
                        ssl_pretrained_dir: str, pretrained_s2G_path: str,
                        i_part: str = "0", all_parts: str = "1") -> None:
        """准备训练数据"""
        print("开始准备训练数据...")
        response = requests.post(f"{self.base_url}/training/prepare", json={
            "exp_name": exp_name,
            "input_text": input_text,
            "input_wav_dir": input_wav_dir,
            "gpu_numbers": gpu_numbers,
            "bert_pretrained_dir": bert_pretrained_dir,
            "ssl_pretrained_dir": ssl_pretrained_dir,
            "pretrained_s2G_path": pretrained_s2G_path,
            "i_part": i_part,
            "all_parts": all_parts
        })
        if response.status_code != 200:
            raise Exception(f"准备训练数据请求失败: {response.text}")
        
        self.current_task_id = response.json()["task_id"]
        self.wait_for_completion()
        print("训练数据准备完成")

    def train_sovits(self, exp_name: str, batch_size: int, total_epoch: int,
                    save_every_epoch: int, gpu_numbers: str,
                    pretrained_s2G: str, pretrained_s2D: str,
                    version: str = "v2",
                    text_low_lr_rate: Optional[float] = 0.4,
                    if_grad_ckpt: Optional[bool] = False,
                    lora_rank: Optional[str] = "32") -> None:
        """训练SoVITS模型
        
        Args:
            exp_name: 实验名称
            batch_size: 批次大小
            total_epoch: 总训练轮数
            save_every_epoch: 每多少轮保存一次
            gpu_numbers: GPU编号
            pretrained_s2G: 预训练SoVITS-G模型路径
            pretrained_s2D: 预训练SoVITS-D模型路径
            version: 模型版本，可选"v1"/"v2"/"v3"，默认为"v2"
            text_low_lr_rate: 文本模块学习率权重，仅v1/v2版本使用
            if_grad_ckpt: 是否开启梯度检查点，仅v3版本使用
            lora_rank: LoRA秩，仅v3版本使用
        """
        print("开始训练SoVITS模型...")
        request_data = {
            "exp_name": exp_name,
            "batch_size": batch_size,
            "total_epoch": total_epoch,
            "version": version,
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": save_every_epoch,
            "gpu_numbers": gpu_numbers,
            "pretrained_s2G": pretrained_s2G,
            "pretrained_s2D": pretrained_s2D
        }
        
        if version in ["v1", "v2"]:
            request_data["text_low_lr_rate"] = text_low_lr_rate
        else:  # v3
            request_data["if_grad_ckpt"] = if_grad_ckpt
            request_data["lora_rank"] = lora_rank
            
        response = requests.post(f"{self.base_url}/training/sovits", json=request_data)
        if response.status_code != 200:
            raise Exception(f"SoVITS训练请求失败: {response.text}")
        
        self.current_task_id = response.json()["task_id"]
        self.wait_for_completion()
        print("SoVITS训练完成")

    def train_gpt(self, exp_name: str, batch_size: int, total_epoch: int,
                 save_every_epoch: int, gpu_numbers: str,
                 pretrained_s1: str) -> None:
        """训练GPT模型"""
        print("开始训练GPT模型...")
        response = requests.post(f"{self.base_url}/training/gpt", json={
            "exp_name": exp_name,
            "batch_size": batch_size,
            "total_epoch": total_epoch,
            "if_dpo": False,
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": save_every_epoch,
            "gpu_numbers": gpu_numbers,
            "pretrained_s1": pretrained_s1
        })
        if response.status_code != 200:
            raise Exception(f"GPT训练请求失败: {response.text}")
        
        self.current_task_id = response.json()["task_id"]
        self.wait_for_completion()
        print("GPT训练完成")

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例
    client = TrainingClient()

    try:
        # 设置路径和参数
        exp_name = "test_APi80"
        input_path = "D:/Desktop/TapiTest/1/int"  # 原始音频文件目录
        output_root = "D:/Desktop/TapiTest/1/output"  # 切分后的音频输出目录
        asr_output_dir = "D:/Desktop/TapiTest/1/asr_output"  # ASR输出目录
        gpu_numbers = "0"  # 使用第一个GPU
        
        # 预训练模型路径
        bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"  # BERT预训练模型
        ssl_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"  # SSL预训练模型
        pretrained_s2G_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"  # SoVITS-G预训练模型
        pretrained_s2D = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"  # SoVITS-D预训练模型
        pretrained_s1 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"  # GPT预训练模型

        # 1. 音频切分
        client.slice_audio(input_path, output_root)

        # 2. 语音识别
        client.run_asr(output_root, asr_output_dir)

        # 3. 准备训练数据
        client.prepare_training(
            exp_name=exp_name,
            input_text=f"{asr_output_dir}/output.list",
            input_wav_dir=output_root,
            gpu_numbers=gpu_numbers,
            bert_pretrained_dir=bert_pretrained_dir,
            ssl_pretrained_dir=ssl_pretrained_dir,
            pretrained_s2G_path=pretrained_s2G_path
        )

        # 4. 训练SoVITS模型 (V2版本)
        client.train_sovits(
            exp_name=exp_name,
            batch_size=4,
            total_epoch=10,
            save_every_epoch=5,
            gpu_numbers=gpu_numbers,
            pretrained_s2G=pretrained_s2G_path,
            pretrained_s2D=pretrained_s2D,
            version="v2",
            text_low_lr_rate=0.4
        )

        # 或者使用V3版本
        # client.train_sovits(
        #     exp_name=exp_name,
        #     batch_size=8,
        #     total_epoch=100,
        #     save_every_epoch=10,
        #     gpu_numbers=gpu_numbers,
        #     pretrained_s2G=pretrained_s2G_path,
        #     pretrained_s2D=pretrained_s2D,
        #     version="v3",
        #     if_grad_ckpt=True,
        #     lora_rank="32"
        # )

        # 5. 训练GPT模型
        client.train_gpt(
            exp_name=exp_name,
            batch_size=4,
            total_epoch=10,
            save_every_epoch=5,
            gpu_numbers=gpu_numbers,
            pretrained_s1=pretrained_s1
        )

        print("所有训练步骤已完成！")

    except Exception as e:
        print(f"训练过程中出错: {e}")