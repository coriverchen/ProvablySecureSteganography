#!/usr/bin/env python

import json
import torch
import numpy as np
import os
from os.path import exists, join, basename, splitext
from flowtron import Flowtron
from data import Data
from scipy.stats import norm
import sys
from functools import reduce
import sys
sys.path.insert(0, 'tacotron2')
sys.path.insert(0, 'tacotron2/waveglow')
from scipy.io.wavfile import write



flowtron_pretrained_model = 'flowtron_ljs.pt'
waveglow_pretrained_model = 'waveglow_256channels_universal_v5.pt'

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

# read config
config = json.load(open('config.json'))
data_config = config["data_config"]
model_config = config["model_config"]
model_config['n_speakers'] = 1 # there are 123 speakers
data_config['training_files'] = 'filelists/ljs_audiopaths_text_sid_train_filelist.txt'
data_config['validation_files'] = data_config['training_files']

# load waveglow
waveglow = torch.load(waveglow_pretrained_model)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval()

# load flowtron
model = Flowtron(**model_config).cuda()
state_dict = torch.load(flowtron_pretrained_model, map_location='cpu')['state_dict']
model.load_state_dict(state_dict)
_ = model.eval()


ignore_keys = ['training_files', 'validation_files']
trainset = Data(data_config['training_files'], **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

def synthesize(speaker_id, text, sigma=0.5, n_frames=500,payload=0.5):
  speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
  text = trainset.get_text(text).cuda()
  speaker_vecs = speaker_vecs[None]
  text = text[None]

  with torch.no_grad():
    residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
    mels, attentions = model.infer(residual, speaker_vecs, text)

  with torch.no_grad():
    audio = waveglow.infer(mels, sigma=1).float()
    audio = audio.detach()
    z , log_s_list, log_det_W_list  = waveglow((mels, audio))
  
  z2 = torch.cuda.FloatTensor(z.shape).normal_()
  # embed message
  with torch.no_grad():
      audio_out = waveglow.reverse_audio_true((mels, z2)) 
      audio_out = audio_out.detach()
      z3 , log_s_list, log_det_W_list  = waveglow((mels, audio_out))

      audio_out = audio_out.cpu().numpy()[0]
      length = z.shape[0]*z.shape[1]*z.shape[2]
      z2 = np.zeros(length)
      meslength = round(len(audio_out)*payload)
      mess= np.random.randint(0,2,meslength)
      mesbit_len = 1
      fenmu = float(2**mesbit_len)

      for i in range(length):
          if i < meslength:
            dec_mes = reduce(lambda a,b: 2*a+b, mess[mesbit_len*i:mesbit_len*(i+1)])
            dec_mes = float(dec_mes)
            while 1:
                z2[i] = np.random.standard_normal()
                if z2[i]>norm.ppf(dec_mes/fenmu) and z2[i]<norm.ppf((dec_mes+1)/fenmu):
                    break 
          else:
            z2[i] = np.random.standard_normal()

      z2 = torch.from_numpy(z2)
      z2 = torch.reshape(z2,z.shape).cuda().float()
      with torch.no_grad():
          audio_out2 = waveglow.reverse_audio_true((mels, z2))
      audio_out2 = audio_out2.cpu().numpy()[0]

#save cover audio and stego audio
  write(os.path.join('./cover', 'sid{}_sigma{}.wav'.format(speaker_id, sigma)),
          data_config['sampling_rate'], audio_out)
  write(os.path.join('./stego', 'sid{}_sigma2{}.wav'.format(speaker_id, sigma)),
          data_config['sampling_rate'], audio_out2)



TEXT = "It is well know that deep generative models have a deep latent space!"
SPEAKER_ID = 0  
synthesize(SPEAKER_ID, TEXT)
