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