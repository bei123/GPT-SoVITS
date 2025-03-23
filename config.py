import sys
import os
import torch

# 推理用的指定模型
sovits_path = ""
gpt_path = ""
is_half_str = os.environ.get("is_half", "True")
is_half = True if is_half_str.lower() == 'true' else False
is_share_str = os.environ.get("is_share", "False")
is_share = True if is_share_str.lower() == 'true' else False

cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

exp_root = "logs"
python_exec = sys.executable or "python"
if torch.cuda.is_available():
    infer_device = "cuda"
else:
    infer_device = "cpu"

webui_port_main = 9874
webui_port_uvr5 = 9873
webui_port_infer_tts = 9872
webui_port_subfix = 9871

api_port = 6006

if infer_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half = False

if(infer_device == "cpu"):
    is_half = False


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
            "ailixiya":"我还有好多好多话想对你说，就这样一直继续下去，好吗？",
            "huahuo":"可聪明的人从一开始就不会入局。你瞧，我是不是更聪明一点？",
            "bufeiyan":"再把精液涂满我的脸，给我自己做一个精液面膜。",
            "guodegang":"哲宗皇帝的妹夫，这个九大王赵佶又是他的小舅子。",
            "hanhong":"谁爱知道，我想什么都没所谓，我所有的一切都是通透的。",
            "zhoujielun":"对，创作不可能无师自通的，我觉得你应该要这样讲啊。",
            "jok":"怎么样才能维持一段良好的关系？在维持一段关系的时候，是不是要",
            "yatuoli":"よしよし、みなも。ほら、見てください。",
            "yangmi":"就是没有什么说特定的，你只是非常喜欢那个现场的氛围",
            "sabeining":"并因此，成功破获了一起公安部督办的毒品大案。",
            "direnjie":"奉天承运，皇帝诏曰，古来圣王治世。",
            "nigemaiti":"亲爱的朋友们，今天的开幕式现场，我们也荣幸的请到了很多尊贵的来宾，他们是。",
            
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
            
        }
