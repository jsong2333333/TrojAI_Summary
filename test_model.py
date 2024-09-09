import os
import torch

MODEL_DIR = '/scratch/data/TrojAI/mitigation-image-classification-jun2024/train-dataset/models/id-00000000'

model = torch.load(os.path.join(MODEL_DIR, 'model.pt'))
data_dir = os.path.join(MODEL_DIR, '')

