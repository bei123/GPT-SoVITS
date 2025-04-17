import os

inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")

if not os.path.exists(pretrained_s2G):
    raise FileNotFoundError(pretrained_s2G)

import torch
import json

# 加载预训练模型的权重以检查版本
pretrained_weights = torch.load(pretrained_s2G, map_location="cpu")
if "version" in pretrained_weights:
    version = pretrained_weights["version"]
else:
    # 如果没有版本信息，则根据模型结构判断
    weight_keys = pretrained_weights["weight"].keys()
    if "text_encoder.emb.weight" in weight_keys:
        emb_shape = pretrained_weights["weight"]["text_encoder.emb.weight"].shape
        if emb_shape[0] == 322:
            version = "v1"
        elif emb_shape[0] == 732:
            version = "v2"
        else:
            version = "v3"
    else:
        # 如果无法判断，使用默认版本
        version = "v2"

print(f"检测到模型版本: {version}")

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import traceback
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import utils

if version != "v3":
    from module.models import SynthesizerTrn
else:
    from module.models import SynthesizerTrnV3 as SynthesizerTrn
from tools.my_utils import clean_path

logging.getLogger("numba").setLevel(logging.WARNING)
# from config import pretrained_s2G

# inp_text=sys.argv[1]
# exp_name=sys.argv[2]
# i_part=sys.argv[3]
# all_parts=sys.argv[4]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[5]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name


hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    
    # 加载配置文件
    hps = utils.get_hparams_from_file(s2config_path)
    
    # 根据版本调整模型参数
    if version == "v1":
        hps.model["text_encoder_embedding_dim"] = 192
        hps.model["text_encoder_num_embeddings"] = 322
    elif version == "v2":
        hps.model["text_encoder_embedding_dim"] = 192
        hps.model["text_encoder_num_embeddings"] = 732
    
    # 创建模型实例
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version=version,
        **hps.model,
    )
    
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()

    # 加载预训练权重
    try:
        missing_keys, unexpected_keys = vq_model.load_state_dict(
            pretrained_weights["weight"], strict=False
        )
        print(f"加载权重成功，缺失的键: {missing_keys}")
        print(f"意外的键: {unexpected_keys}")
    except Exception as e:
        print(f"加载权重时出错: {str(e)}")
        raise

    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name = clean_path(wav_name)
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))

# 在脚本末尾添加以下代码，创建一个不带i_part后缀的链接文件
if os.path.exists(semantic_path):
    # 创建一个不带i_part后缀的链接文件
    link_path = "%s/6-name2semantic.tsv" % (opt_dir)
    if not os.path.exists(link_path):
        # 如果文件不存在，则创建一个链接
        with open(semantic_path, 'r', encoding='utf-8') as src, open(link_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
