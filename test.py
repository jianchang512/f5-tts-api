import requests

res=requests.post('http://127.0.0.1:5010/api',data={
    "gen_text": '你好，今天是个不错的日子',
    "ref_text": '你说四大皆空，却为何紧闭双眼，若你睁开眼睛看看我，我不相信你，两眼空空',
    "model": 'f5-tts'
},files={"audio":open('./1.wav','rb')})

if res.status_code!=200:
    print(res.text)
    exit()

with open("ceshi.wav",'wb') as f:
    f.write(res.content)
    
    
    