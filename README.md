# F5-TTS-api

这是用于 [F5-TTS](https://github.com/SWivid/F5-TTS) 项目的api和webui，fork自 [ThisModernDay](https://github.com/ThisModernDay/f5-tts)

> F5-TTS是由上海交通大学开源的一款基于流匹配的全非自回归文本到语音转换系统（Text-to-Speech，TTS）。它以其高效、自然和多语言支持的特点脱颖而出

## 功能

- 提供api接口文件 `api.py`，可对接于视频翻译项目 [pyvideotrans](https://github.com/jianchang512/pyvideotrans)
- 提供webui界面，可在浏览器中进行声音克隆
- 提供windows下整合包

## 源码部署

0. 需要提前安装 `python3.10`  `git`

1. 克隆这个仓库:   

```
   git clone https://github.com/jianchang512/f5-tts-api
   cd f5-tts   
```

2. 创建虚拟环境

```
	 python -m venv venv
```

windows下执行命令激活环境 `.\venv\scripts\activate`

Mac/Linux下执行命令激活环境 `. venv/bin/activate`

3. 安装依赖:   ` pip install -r requirements.txt  `

4. 启动api服务:   `python api.py`，接口地址是 `http://127.0.0.1:5010/api`

5. 启动webui服务  `python webui.py`,在浏览器中打开  `http://localhost:5000`.


## 整合包部署5G(包含f5-tts模型和medium模型及环境)

整合包下载地址 123网盘:

> 整合包仅可用于 Windows10/11, 下载后解压即用

1. 启动Api服务:  双击 `run-api.bat`，接口地址是 `http://127.0.0.1:5010/api`

2. 启动webui服务：双击 `run-webui.bat`, 自动打开浏览器页面  `http://localhost:5000`.

>
> 整合包默认不支持GPU加速，如果需要GPU加速， 请确认拥有英伟达显卡，并且已安装配置好CUDA环境，然后在当前api.py所在文件夹内地址栏中，输入`cmd`回车，在打开的终端窗口中分别执行下面2条命令
>
> `pip uninstall -y torch torchaudio`
>
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`
>



## 使用注意/代理VPN

1. 声音克隆中，需要用到 asr 模型，默认使用 faster-whisper 的 medium 模型，该模型不是效果最佳的，若需提升效果，可修改为`large`模型，打开 `api.py`文件和`webui.py`文件，找到第一行，将 `medium`修改为`large`，注意large模型对计算机资源要求较高，如果设备性能不佳，可能会卡死或非常缓慢

2. asr模型需要从 `huggingface.co`网站在线下载，该站点无法在国内访问，如果你是源码部署，或者整合包部署但修改了默认模型`medium`，将会从 `huggingface.co`在线下载，请提前开启系统代理或全局代理，否则模型会下载失败



## 在视频翻译软件中使用

1. 启动api服务
2. 打开视频翻译软件，找到菜单-TTS设置-F5-TTS，填写api地址，如果未修改过地址，填写`http://127.0.0.1:5010`
3. 填写参考音频和音频内文本

![](https://pyvideotrans.com/img/f5002.jpg)




## API 使用示例

```
import requests

res=requests.post('http://127.0.0.1:5010/api',data={
    "ref_text": '古老星系中发现了有机分子，我们离第三类接触还有多远呢',
    "gen_text": '今天是个好日子，外面下了大暴雨，海水也冲上了岸。',
    "model": 'f5-tts'
},files={"audio":open('c:/users/c1/videos/5s.wav','rb')})

if res.status_code!=200:
    print(res.text)
    exit()

with open("ceshi.wav",'wb') as f:
    f.write(res.content)
```

