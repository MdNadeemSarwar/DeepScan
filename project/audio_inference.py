import torch
import numpy as np
from torch import Tensor
import yaml
import librosa
from model import RawNet
from torch.nn import functional as F

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_sample(sample_path, max_len=96000):
    y_list = []
    y, sr = librosa.load(sample_path, sr=None)
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)
    if len(y) <= 96000:
        return [Tensor(pad(y, max_len))]
    for i in range(int(len(y) / 96000)):
        y_seg = y[i * 96000: (i + 1) * 96000]
        y_pad = pad(y_seg, max_len)
        y_list.append(Tensor(y_pad))
    return y_list

# Load model once
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('model_config_RawNet.yaml', 'r') as f:
    config = yaml.safe_load(f)

audio_model = RawNet(config['model'], device)
audio_model = audio_model.to(device)
audio_model.load_state_dict(torch.load(
    'model/librifake_pretrained_lambda0.5_epoch_25.pth',
    map_location=device
))
audio_model.eval()

def predict_audio(audio_path):
    try:
        samples = load_sample(audio_path)
        out_list = []
        for m_batch in samples:
            m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
            logits, _ = audio_model(m_batch)
            probs = F.softmax(logits, dim=-1)
            out_list.append(probs.tolist()[0])
        
        result = np.average(out_list, axis=0).tolist()
        fake_prob = result[0]
        real_prob = result[1]
        
        label = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = round(max(fake_prob, real_prob) * 100, 2)
        
        return label, confidence
    except Exception as e:
        return None, str(e)