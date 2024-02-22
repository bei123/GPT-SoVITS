import sys
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QWidget, QFormLayout, QLineEdit, QPushButton,
                             QFileDialog, QLabel, QMessageBox, QComboBox, QVBoxLayout)
import requests
import re
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：]'
    items = re.split(f'({punds})', inp)
    items = ["".join(group) for group in zip(items[::2], items[1::2])]
    opt = "\n".join(items)
    return opt

class TextToSpeechApp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.serverUrlEdit = QLineEdit("http://127.0.0.1")
        self.serverPortEdit = QLineEdit("9880")
        self.topKEdit = QLineEdit("20")
        self.topPEdit = QLineEdit("0.6")
        self.temperatureEdit = QLineEdit("0.6")
        self.setWindowTitle('GPT-Sovits API调用')
        # self.setWindowIcon(QIcon('path/to/your/icon.png'))  # 设置窗口图标
        self.resize(600, 500)  # 调整窗口大小
        self.initUI()
        # self.get_and_display_files()  # 调用该方法以填充文件列表
        # self.getSovitsModelFiles()
        # self.getGPTModelFiles()  # 在初始化时获取 Sovits 模型文件列表
        
        self.formLayout.addRow(QLabel('服务器地址：'), self.serverUrlEdit)
        self.formLayout.addRow(QLabel('服务器端口：'), self.serverPortEdit)

    
    def refreshModelPaths(self):
        self.getGPTModelFiles()  # 刷新GPT模型文件列表
        self.getSovitsModelFiles()  # 刷新Sovits模型文件列表



    def uploadFileToServervideo(self):
        # 从输入字段获取服务器地址和端口
        server_url = self.serverUrlEdit.text().strip()
        server_port = self.serverPortEdit.text().strip()
        full_url = f"{server_url}:{server_port}/upload_video"
        
        # 使用这些设置发送网络请求
        try:
            filePath, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*)")
            if filePath:  # 确保用户选择了文件
                files = {'file': open(filePath, 'rb')}
                response = requests.post(full_url, files=files)
                if response.status_code == 200:
                    self.showMessageBox('成功', f'文件上传成功: {response.json()["message"]}')
                else:
                    self.showMessageBox('错误', f'文件上传失败: {response.json()["message"]}')
        except Exception as e:
            self.showMessageBox('错误', f'上传过程中发生异常: {str(e)}')
        finally:
            if 'files' in locals():
                files['file'].close()

                    
    
        


    def showMessageBox(self, title, message):
        QMessageBox.information(self, title, message)


    def initUI(self):
        self.isUpdatingModel = False
        self.setStyleSheet("""
        QWidget {
            font-size: 16px; /* 增加基础字体大小 */
        }
        QLineEdit, QComboBox {
            border: 2px solid #c4c4c4; /* 加粗边框 */
            border-radius: 6px; /* 增加边框圆角 */
            padding: 8px; /* 增加填充以增大输入框高度 */
            min-height: 40px; /* 增加最小高度 */
            font-size: 16px; /* 增加字体大小以改善可读性 */
        }
        QPushButton {
            background-color: #007BFF;
            color: white;
            border-radius: 6px; /* 增加按钮圆角 */
            padding: 10px 15px; /* 增加按钮内边距 */
            min-width: 100px; /* 增加按钮最小宽度 */
            min-height: 40px; /* 增加按钮最小高度 */
            font-size: 18px; /* 增加按钮字体大小 */
            border: none; /* 移除边框 */
        }
        QPushButton:hover {
            background-color: #0056b3; /* 悬停时的背景颜色 */
        }
        QPushButton:pressed {
            background-color: #004494; /* 按下时的背景颜色 */
        }
        QLabel {
            qproperty-alignment: 'AlignVCenter'; /* 确保标签垂直居中 */
            font-size: 16px; /* 标签字体大小 */
        }
        QFormLayout::label {
            min-width: 180px; /* 设置标签的最小宽度，确保对齐 */
        }
    
        }
    """)


    # 主布局
        self.layout = QVBoxLayout(self)

    # 表单布局
        self.formLayout = QFormLayout()
        self.layout.addLayout(self.formLayout)
        self.sovitsModelComboBox = QComboBox()
        self.gptModelComboBox = QComboBox()
        self.formLayout.addRow(QLabel('Top K：'), self.topKEdit)
        self.formLayout.addRow(QLabel('Top P：'), self.topPEdit)
        self.formLayout.addRow(QLabel('Temperature：'), self.temperatureEdit)

    # 添加控件到表单布局
        self.setupFormFields()

    # 按钮布局
        self.setupButtonLayout()

    def setupFormFields(self):
    # 设置表单字段
        self.textFilePath = self.createLineItem("文本文件路径：", "选择文件", "Text Files (*.txt)")
        self.referAudioComboBox = QComboBox()
        self.promptText = QLineEdit()
        self.promptLanguage = QLineEdit()
        self.promptLanguage = QComboBox()
        self.promptLanguage.addItems(['中文', '日文', '英文', '中英混合', '日英混合', '多语种混合'])
        self.textLanguage = QLineEdit()
        self.textLanguage = QComboBox()
        self.textLanguage.addItems(['中文', '日文', '英文', '中英混合', '日英混合', '多语种混合'])
        self.outputFilePath = self.createLineItem("保存推理音频文件路径：", "保存合成音频文件路径", "Audio Files (*.wav)", save=True)
        self.cutMethodComboBox = QComboBox()
        self.cutMethodComboBox.addItems(['cut1(凑四句一切)', 'cut2(凑五十字一切)', 'cut3(按中文句号。切割)', 'cut4(按英文.句号切割)', 'cut5(按标点符号切割)'])

    # 添加字段到表单布局
        fields = [
            ('文本文件路径：', self.textFilePath),
            ('参考音频文件路径：', self.referAudioComboBox),
            ('提示文本：', self.promptText),
            ('提示文本的语言：', self.promptLanguage),
            ('要转换文本的语言：', self.textLanguage),
            ('保存推理音频文件路径：', self.outputFilePath),
            ('选择切割方法：', self.cutMethodComboBox)
        ]
        for label, widget in fields:
            self.formLayout.addRow(QLabel(label), widget)

    def setupButtonLayout(self):
    # 按钮水平布局
        buttonLayout = QHBoxLayout()

    # 创建按钮
        self.refreshButton = QPushButton('刷新参考音频文件列表')
        self.submitButton = QPushButton('开始推理')
        self.updateModelPathsButton = QPushButton('更换模型', self)
        uploadFileButton = QPushButton('上传参考音频文件')
        self.refreshModelPathsButton = QPushButton('刷新模型列表')
        self.initializeButton = QPushButton('连接服务器(第一步必做！！！)')
        
        

    # 添加按钮到布局
        buttonLayout.addWidget(self.refreshButton)
        buttonLayout.addWidget(uploadFileButton)
        buttonLayout.addWidget(self.updateModelPathsButton)
        buttonLayout.addWidget(self.submitButton)
        buttonLayout.addWidget(self.refreshModelPathsButton)
        
        self.formLayout.addRow(self.initializeButton)  # 根据您的布局需求添加到界面中
        self.initializeButton.clicked.connect(self.initializeConnection)
        

    # 将按钮布局添加到主布局
        self.layout.addLayout(buttonLayout)

    # 连接按钮信号
        self.refreshButton.clicked.connect(self.get_and_display_files)
        self.submitButton.clicked.connect(self.submitForm)
        uploadFileButton.clicked.connect(self.uploadFileToServervideo)
        self.updateModelPathsButton.clicked.connect(self.updateModelPaths)
        self.formLayout.addRow(QLabel('GPT模型文件：'), self.gptModelComboBox)
        self.formLayout.addRow(QLabel('Sovits模型文件：'), self.sovitsModelComboBox)
        self.refreshModelPathsButton.clicked.connect(self.refreshModelPaths)


    def initializeConnection(self):
    # 从输入字段获取服务器地址和端口
        server_url = self.serverUrlEdit.text().strip()
        server_port = self.serverPortEdit.text().strip()
        if not server_url or not server_port:
            QMessageBox.warning(self, '警告', '服务器地址和端口不能为空。')
            return
    # 尝试获取文件列表和模型文件
        self.get_and_display_files()
        self.getSovitsModelFiles()
        self.getGPTModelFiles()

    def getGPTModelFiles(self):
    # 从输入字段获取服务器地址和端口
        server_url = self.serverUrlEdit.text().strip()
        server_port = self.serverPortEdit.text().strip()
        full_url = f"{server_url}:{server_port}/GPT_list_files"

        try:
            response = requests.get(full_url)
            if response.status_code == 200:
                files = response.json()
                self.gptModelComboBox.clear()
                self.gptModelComboBox.addItems(files)
            else:
                QMessageBox.critical(self, '失败', f'无法获取 GPT 模型文件列表: {response.text}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'请求发送异常: {str(e)}')

    def getSovitsModelFiles(self):
    # 从输入字段获取服务器地址和端口
        server_url = self.serverUrlEdit.text().strip()
        server_port = self.serverPortEdit.text().strip()
        full_url = f"{server_url}:{server_port}/SOVITS_list_files"

        try:
            response = requests.get(full_url)
            if response.status_code == 200:
                files = response.json()
                self.sovitsModelComboBox.clear()
                self.sovitsModelComboBox.addItems(files)
            else:
                QMessageBox.critical(self, '失败', f'无法获取 Sovits 模型文件列表: {response.text}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'请求发送异常: {str(e)}')


    def updateModelPaths(self):
        if self.isUpdatingModel:
            return  # 如果正在更新，则直接返回，不执行后续操作
        self.isUpdatingModel = True
        gpt_path = self.gptModelComboBox.currentText()
        sovits_path = self.sovitsModelComboBox.currentText()
        data = {
            "gpt_model_path": gpt_path,
            "sovits_model_path": sovits_path
        }
        # 从输入字段获取服务器地址和端口
        server_url = self.serverUrlEdit.text().strip()
        server_port = self.serverPortEdit.text().strip()
        full_url = f"{server_url}:{server_port}/set_model"

        try:
            response = requests.post(full_url, json=data)
            if response.status_code == 200:
                QMessageBox.information(self, '成功', '模型路径更新成功')
                self.refreshModelPaths()  # 刷新模型列表
            else:
                QMessageBox.critical(self, '失败', f'模型路径更新失败: {response.text}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'请求发送异常: {str(e)}')
        finally:
            self.isUpdatingModel = False  # 无论成功还是失败最后都应该重置标志
            



    def get_and_display_files(self):
        self.referAudioComboBox.clear()  # 清除现有条目
        # 从输入字段获取服务器地址和端口
        server_url = self.serverUrlEdit.text().strip()
        server_port = self.serverPortEdit.text().strip()
        full_url = f"{server_url}:{server_port}/video_list_files"

        try:
            response = requests.get(full_url)
            if response.status_code == 200:
                files = response.json()
                for file_path in files:
                    file_name = file_path.split('/')[-1]
                    self.referAudioComboBox.addItem(file_name, file_path)
            else:
                QMessageBox.critical(self, '错误', '无法从服务器获取文件列表。')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'获取文件列表时发生错误: {e}')



    def updateReferAudioPath(self):
       
        selected_file = self.referAudioComboBox.currentText()
        # self.referWavPath.setText(selected_file)  

    def createFormLayout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('文本文件路径：'), self.createBrowseLayout(self.textFilePath, "Text Files (*.txt)"))
        layout.addRow(QLabel('参考音频文件路径：'), self.createBrowseLayout(self.referWavPath, "Audio Files (*.wav;*.mp3)"))
        layout.addRow(QLabel('提示文本：'), self.promptText)
        layout.addRow(QLabel('提示文本的语言：'), self.promptLanguage)
        layout.addRow(QLabel('要转换文本的语言：'), self.textLanguage)
        layout.addRow(QLabel('输出音频文件路径：'), self.createBrowseLayout(self.outputFilePath, "Audio Files (*.wav)", save=True))
        layout.addRow(QLabel('选择切割方法：'), self.cutMethodComboBox)
        layout.addRow(self.submitButton)
        
        return layout

    def createBrowseLayout(self, lineEdit, fileType="All Files (*)", save=False):
        button = QPushButton('浏览')
        button.clicked.connect(lambda: self.browseFile(lineEdit, fileType, save))
        browseLayout = QVBoxLayout()
        browseLayout.addWidget(lineEdit)
        browseLayout.addWidget(button)
        return browseLayout

    def browseFile(self, lineEdit, fileType, save):
        if save:
            path, _ = QFileDialog.getSaveFileName(self, "文件保存为", "", fileType)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", fileType)
        if path:
            lineEdit.setText(path)

    def submitForm(self):
    # 提交表单逻辑
        try:
        # 从文本文件路径中读取内容
            text_content = self.readAndCleanText(self.textFilePath.text())
            processed_text = self.process_text_with_cut_method(text_content, self.cutMethodComboBox.currentText())

        # 从下拉框中获取选定的音频文件路径
            refer_wav_path_index = self.referAudioComboBox.currentIndex()  # 获取当前选中项的索引
            refer_wav_path = self.referAudioComboBox.itemData(refer_wav_path_index)  # 使用索引检索完整路径

        # 验证是否选择了文件路径
            if not refer_wav_path:
                QMessageBox.critical(self, '错误', '请先选择一个参考音频文件。')
                return

        
            top_k = int(self.topKEdit.text())
            top_p = float(self.topPEdit.text())
            temperature = float(self.temperatureEdit.text())

        
            data = {
                "refer_wav_path": refer_wav_path,  
                "prompt_text": self.promptText.text(),
                "prompt_language": self.promptLanguage.currentText(),
                "text": processed_text,
                "text_language": self.textLanguage.currentText(),
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            }

        # 向服务器发送请求
            self.postTextToTtsApi(data, self.outputFilePath.text())
        except Exception as e:
            QMessageBox.critical(self, '错误', f'处理文件时发生错误: {e}')


    def readAndCleanText(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return re.sub(r'\s+', '', file.read().strip())

    def process_text_with_cut_method(self, text, method):
        # 根据用户选择的cut方法处理文本
        cut_methods = {'cut1(凑四句一切)': cut1, 'cut2(凑五十字一切)': cut2, 'cut3(按中文句号。切割)': cut3, 'cut4(按英文.句号切割)': cut4, 'cut5(按标点符号切割)': cut5}
        return cut_methods[method](text)

    def postTextToTtsApi(self, data, output_file_path):
    # 从输入字段获取服务器地址和端口
        server_url = self.serverUrlEdit.text().strip()
        server_port = self.serverPortEdit.text().strip()
    # 构建完整的URL
        full_url = f"{server_url}:{server_port}/"  # 如果API有具体的路径，需要将"/"替换成"/api_path"

        try:
            response = requests.post(full_url, json=data)
            if response.status_code == 200:
                with open(output_file_path, "wb") as f:
                    f.write(response.content)
                QMessageBox.information(self, '成功', f'音频已成功保存至 {output_file_path}')
            else:
                QMessageBox.critical(self, '错误', f'请求失败，状态码：{response.status_code}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'请求发送异常: {str(e)}')

    def createLineItem(self, label, buttonText, fileType, save=False):
        
        lineEdit = QLineEdit()
        button = QPushButton(buttonText)
        button.clicked.connect(lambda: self.browseFile(lineEdit, fileType, save))
        rowLayout = QHBoxLayout()
        rowLayout.addWidget(lineEdit)
        rowLayout.addWidget(button)
        self.formLayout.addRow(QLabel(label), rowLayout)
        return lineEdit
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TextToSpeechApp()
    ex.show()
    sys.exit(app.exec_())
