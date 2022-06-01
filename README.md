# KSS-TTS
[KSS Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)을 활용한 TTS 모델 웹서비스 - [ACCESS URL](https://tts.favorcat.dev)

- Model : [Tacotron2](https://github.com/favorcat/Tacotron-Korean-Tensorflow2)
- [FastAPI](https://fastapi.tiangolo.com/ko/)
- [Uvicorn](https://www.uvicorn.org/)

---
### 서버 실행
```
uvicorn app:app --host '0.0.0.0' --reload
```
```
uvicorn app:app --host '0.0.0.0' --ssl-keyfile './private_key.pem' --ssl-certfile './server_crt.pem' --reload
```
---
### codepen
- [wave](https://codepen.io/rachelmcgrane/pen/VexWdX)
- [Download button](https://codepen.io/aaroniker/pen/KjJQER)
- [Text editor](https://codepen.io/shotastage/pen/KaKwya)