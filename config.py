import os
import re
import sys

import torch

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language=os.environ.get("language", "Auto"))


pretrained_sovits_name = {
    "v1": "GPT_SoVITS/pretrained_models/s2G488k.pth",
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "v3": "GPT_SoVITS/pretrained_models/s2Gv3.pth",  ###v3v4还要检查vocoder，算了。。。
    "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
}

pretrained_gpt_name = {
    "v1": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "v3": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v4": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2Pro": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
}
name2sovits_path = {
    # i18n("不训练直接推v1底模！"): "GPT_SoVITS/pretrained_models/s2G488k.pth",
    i18n("不训练直接推v2底模！"): "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    # i18n("不训练直接推v3底模！"): "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    # i18n("不训练直接推v4底模！"): "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    i18n("不训练直接推v2Pro底模！"): "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    i18n("不训练直接推v2ProPlus底模！"): "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
}
name2gpt_path = {
    # i18n("不训练直接推v1底模！"):"GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    i18n(
        "不训练直接推v2底模！"
    ): "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    i18n("不训练直接推v3底模！"): "GPT_SoVITS/pretrained_models/s1v3.ckpt",
}
SoVITS_weight_root = [
    "SoVITS_weights",
    "SoVITS_weights_v2",
    "SoVITS_weights_v3",
    "SoVITS_weights_v4",
    "SoVITS_weights_v2Pro",
    "SoVITS_weights_v2ProPlus",
]
GPT_weight_root = [
    "GPT_weights",
    "GPT_weights_v2",
    "GPT_weights_v3",
    "GPT_weights_v4",
    "GPT_weights_v2Pro",
    "GPT_weights_v2ProPlus",
]
SoVITS_weight_version2root = {
    "v1": "SoVITS_weights",
    "v2": "SoVITS_weights_v2",
    "v3": "SoVITS_weights_v3",
    "v4": "SoVITS_weights_v4",
    "v2Pro": "SoVITS_weights_v2Pro",
    "v2ProPlus": "SoVITS_weights_v2ProPlus",
}
GPT_weight_version2root = {
    "v1": "GPT_weights",
    "v2": "GPT_weights_v2",
    "v3": "GPT_weights_v3",
    "v4": "GPT_weights_v4",
    "v2Pro": "GPT_weights_v2Pro",
    "v2ProPlus": "GPT_weights_v2ProPlus",
}


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split("(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def get_weights_names():
    SoVITS_names = []
    for key in name2sovits_path:
        if os.path.exists(name2sovits_path[key]):
            SoVITS_names.append(key)
    for path in SoVITS_weight_root:
        if not os.path.exists(path):
            continue
        for name in os.listdir(path):
            if name.endswith(".pth"):
                SoVITS_names.append("%s/%s" % (path, name))
    if not SoVITS_names:
        SoVITS_names = [""]
    GPT_names = []
    for key in name2gpt_path:
        if os.path.exists(name2gpt_path[key]):
            GPT_names.append(key)
    for path in GPT_weight_root:
        if not os.path.exists(path):
            continue
        for name in os.listdir(path):
            if name.endswith(".ckpt"):
                GPT_names.append("%s/%s" % (path, name))
    SoVITS_names = sorted(SoVITS_names, key=custom_sort_key)
    GPT_names = sorted(GPT_names, key=custom_sort_key)
    if not GPT_names:
        GPT_names = [""]
    return SoVITS_names, GPT_names


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": SoVITS_names, "__type__": "update"}, {
        "choices": GPT_names,
        "__type__": "update",
    }


# 推理用的指定模型
sovits_path = ""
gpt_path = ""
is_half_str = os.environ.get("is_half", "True")
is_half = True if is_half_str.lower() == "true" else False
is_share_str = os.environ.get("is_share", "False")
is_share = True if is_share_str.lower() == "true" else False

cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

exp_root = "logs"
python_exec = sys.executable or "python"

webui_port_main = 9874
webui_port_uvr5 = 9873
webui_port_infer_tts = 9872
webui_port_subfix = 9871

api_port = 6006


# Thanks to the contribution of @Karasukaigan and @XXXXRT666
def get_device_dtype_sm(idx: int) -> tuple[torch.device, torch.dtype, float, float]:
    cpu = torch.device("cpu")
    cuda = torch.device(f"cuda:{idx}")
    if not torch.cuda.is_available():
        return cpu, torch.float32, 0.0, 0.0
    device_idx = idx
    capability = torch.cuda.get_device_capability(device_idx)
    name = torch.cuda.get_device_name(device_idx)
    mem_bytes = torch.cuda.get_device_properties(device_idx).total_memory
    mem_gb = mem_bytes / (1024**3) + 0.4
    major, minor = capability
    sm_version = major + minor / 10.0
    is_16_series = bool(re.search(r"16\d{2}", name)) and sm_version == 7.5
    if mem_gb < 4 or sm_version < 5.3:
        return cpu, torch.float32, 0.0, 0.0
    if sm_version == 6.1 or is_16_series == True:
        return cuda, torch.float32, sm_version, mem_gb
    if sm_version > 6.1:
        return cuda, torch.float16, sm_version, mem_gb
    return cpu, torch.float32, 0.0, 0.0


IS_GPU = True
GPU_INFOS: list[str] = []
GPU_INDEX: set[int] = set()
GPU_COUNT = torch.cuda.device_count()
CPU_INFO: str = "0\tCPU " + i18n("CPU训练,较慢")
tmp: list[tuple[torch.device, torch.dtype, float, float]] = []
memset: set[float] = set()

for i in range(max(GPU_COUNT, 1)):
    tmp.append(get_device_dtype_sm(i))

for j in tmp:
    device = j[0]
    memset.add(j[3])
    if device.type != "cpu":
        GPU_INFOS.append(f"{device.index}\t{torch.cuda.get_device_name(device.index)}")
        GPU_INDEX.add(device.index)

if not GPU_INFOS:
    IS_GPU = False
    GPU_INFOS.append(CPU_INFO)
    GPU_INDEX.add(0)

infer_device = max(tmp, key=lambda x: (x[2], x[3]))[0]
is_half = any(dtype == torch.float16 for _, dtype, _, _ in tmp)


class Config:
    def __init__(self):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.is_half = is_half

        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.pretrained_sovits_path = pretrained_sovits_path
        self.pretrained_gpt_path = pretrained_gpt_path

        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device

        self.webui_port_main = webui_port_main
        self.webui_port_uvr5 = webui_port_uvr5
        self.webui_port_infer_tts = webui_port_infer_tts
        self.webui_port_subfix = webui_port_subfix

        self.api_port = api_port
        
        
        self.modelToPromptText = {
            "wendi": "他曾经与我一同聆听风的歌唱，一同弹奏蒲公英的诗篇",
            "shenglilinghua": "这里有别于神里家的布景，移步之间，处处都有新奇感。",
            "paimeng": "既然罗莎莉亚说足迹上有元素力，用元素视野应该能很清楚地看到吧。",
            "leidianjiangjun":"我此番也是受神子之邀，体验一下市井游乐的氛围，和各位并无二致。",
            "keli":"买东西那天也有一个人帮了开了款式，那个人好像叫",
            "hutao":"本堂主略施小计，你就败下阵来了，嘿嘿。",
            "ganyu":"但只要最后落在具体的人身上，那，我可以想办法。",
            "funingna":"太普通了！哼，这种缺乏特色的料理得不到我的认可！",
            "bachongshenzi":"在这姑且属于人类的社会里，我也不过凭自己兴趣照做而已",
            "ailixiya":"就这样一直继续下去好吗？这么早起床是为了能早点见到我吗？",
            "huahuo":"可聪明的人从一开始就不会入局。你瞧，我是不是更聪明一点？",
            "bufeiyan":"再把精液涂满我的脸，给我自己做一个精液面膜。",
            "guodegang":"哲宗皇帝的妹夫，这个九大王赵佶又是他的小舅子。",
            "hanhong":"谁爱知道，我想什么都没所谓，我所有的一切都是通透的。",
            "zhoujielun":"对，创作不可能无师自通的，我觉得你应该要这样讲啊。",
            "jok":"怎么样才能维持一段良好的关系？在维持一段关系的时候，是不是要",
            "yatuoli":"よしよし、みなも。ほら、見てください。",
            "yangmi":"变得所谓的不好，是一个人的原因或者是一件事的原因。",
            "sabeining":"并因此，成功破获了一起公安部督办的毒品大案。",
            "direnjie":"奉天承运，皇帝诏曰，古来圣王治世。",
            "nigemaiti":"亲爱的朋友们，今天的开幕式现场，我们也荣幸的请到了很多尊贵的来宾，他们是。",
            "yueyunpeng":"哎呦，您说的对是我记错了，尚九熙确实是师傅郭德纲的徒弟。",
            "ruoruo":"别看我穷,我可是掌握了一百种犒劳自己的理由。",
            "xinxiaomeng":"一直病到了月尾,然后我整整一个月,我都没有办法好好的录视频,我就一直在鸽",
            "jiazi":"刚逛超市去了，买了这么一点。",
            "km":"だったんですけどあの一つねファッションショーじゃなくて",
            "qiyana":"突然好想吃美少女味的泡面啊！",
            "buluoniya":"托帕小姐，如果你需要一个舒适的地方落脚。",
            "wzxq":"家人们我现实中追我的人都一堆了,我要用别人的照片跟别人网恋,我一直在做符合我体量该做的事情"
            
        }

       
        self.modelToPromptLanguage = {
            "wendi": "zh",  
            "shenglilinghua": "zh",  
            "paimeng": "zh",  
            "keli":"zh",
            "leidianjiangjun":"zh",
            "hutao":"zh",
            "ganyu":"zh",
            "funingna":"zh",
            "bachongshenzi":"zh",
            "ailixiya":"zh",
            "huahuo":"zh",
            "bufeiyan":"zh",
            "guodegang":"zh",
            "zhoujielun":"zh",
            "jok":"zh",
            "yatuoli":"ja",
            "hanhong":"zh",
            "yangmi":"zh",
            "sabeining":"zh",
            "direnjie":"zh",
            "nigemaiti":"zh",
            "yueyunpeng":"zh",
            "ruoruo":"zh",
            "xinxiaomeng":"zh",
            "jiazi":"zh",
            "km":"ja",
            "qiyana":"zh",
            "buluoniya":"zh",
            "wzxq":"zh"
            
        }

