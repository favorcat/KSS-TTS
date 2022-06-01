import numpy as np
from fastapi import FastAPI, Form, Request
import pandas as pd
from starlette.responses import HTMLResponse 
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from os import getcwd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

import os, librosa, glob, scipy
import soundfile as sf
import matplotlib.pyplot as plt 
from jamo import hangul_to_jamo
from tensorflow.keras.optimizers import Adam
from models.tacotron import Tacotron
from models.tacotron import post_CBHG
from models.modules import griffin_lim
from util.hparams import *
from util.plot_alignment import plot_alignment
from util.text import sequence_to_text, text_to_sequence
from keras.models import load_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class TextArea(BaseModel):
    content: str
    
@app.get('/', response_class=HTMLResponse) #data input by forms
def form_view(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})

@app.post('/') #prediction on data
def predict(textIn:list = Form(...)): #input is from forms
    text = list(textIn)
    print(text)
    save_dir = './output'
    os.makedirs(save_dir, exist_ok=True)
    
    f = open('./output/cnt.txt', "r")
    cnt = f.readline()
    cnt = int(cnt)
    cnt += 1
    name_file = str(cnt) + ".wav"
    f.close()
    f = open('./output/cnt.txt', "w")
    f.write(str(cnt))
    f.close()
        
    # mel-spectrogram 생성
    def test_step(txt):
        max_iter = 50
        seq = text_to_sequence(txt)
        enc_input = np.asarray([seq], dtype=np.int32)
        sequence_length = np.asarray([len(seq)], dtype=np.int32)
        dec_input = np.zeros((1, max_iter, mel_dim), dtype=np.float32)

        pred = []
        for i in range(1, max_iter+1):
            mel_out, alignment = model(enc_input, sequence_length, dec_input, is_training=False)
            if i < max_iter:
                dec_input[:, i, :] = mel_out[:, reduction * i - 1, :]
            pred.extend(mel_out[:, reduction * (i-1) : reduction * i, :])

        pred = np.reshape(np.asarray(pred), [-1, mel_dim])
        alignment = np.squeeze(alignment, axis=0)

        np.save(os.path.join(save_dir, 'mel-{}'.format(cnt)), pred, allow_pickle=False)

        input_seq = sequence_to_text(seq)
        alignment_dir = os.path.join(save_dir, 'align-{}.png'.format(cnt))
        plot_alignment(alignment, alignment_dir, input_seq)

    model = Tacotron(K=16, conv_dim=[128, 128])
    checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint.restore(tf.train.latest_checkpoint('train1.h5')).expect_partial()
    checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/1')).expect_partial()
    
    for i, text in enumerate(text):
        jamo = ''.join(list(hangul_to_jamo(text)))
        test_step(jamo)

    # 음성 파일 생성
    mel_filename = 'mel-' + str(cnt) + '.npy'
    mel_list = glob.glob(os.path.join(save_dir, mel_filename))
    def test2_step(mel, idx):
        mel = np.expand_dims(mel, axis=0)
        pred = model(mel, is_training=False)

        pred = np.squeeze(pred, axis=0)
        pred = np.transpose(pred)

        pred = (np.clip(pred, 0, 1) * max_db) - max_db + ref_db
        pred = np.power(10.0, pred * 0.05)
        wav = griffin_lim(pred ** 1.5)
        wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
        wav = librosa.effects.trim(wav, frame_length=win_length, hop_length=hop_length)[0]
        wav = wav.astype(np.float32)
        sf.write(os.path.join(save_dir, '{}.wav'.format(cnt)), wav, sample_rate)
        

    model = post_CBHG(K=8, conv_dim=[256, mel_dim])
    optimizer = Adam()
    step = tf.Variable(0)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=step)
    checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/2')).expect_partial()

    for i, fn in enumerate(mel_list):
        mel = np.load(fn)
        test2_step(mel, i)
    
    print(text)
    return FileResponse(path= getcwd() +"/output/" + name_file, media_type='application/octet-stream', filename=name_file)