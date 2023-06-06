from EEGClip.clip_models import EEGClipModel
import torch

model_name = "/home/jovyan/EEGClip/results/wandb/EEGClip/df7e5wqd/checkpoints/epoch=7-step=48696.ckpt"

eegclipmodel = EEGClipModel.load_from_checkpoint(model_name)
EEGEncoder = torch.nn.Sequential(eegclipmodel.eeg_encoder,eegclipmodel.eeg_projection)
'''
for key in list(eegclipmodel.state_dict().keys()):
    if "text_encoder" in key:
        print(key)
        del eegclipmodel.state_dict()[key]
#print(eegclipmodel.state_dict().keys())
'''
projectionhead = list(EEGEncoder.children())[-1]
layer_sizes = []
for layer in projectionhead.children():
    if hasattr(layer, 'out_features'):
        layer_sizes.append(layer.out_features)
print(layer_sizes)