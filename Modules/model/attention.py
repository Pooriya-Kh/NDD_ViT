'''
The module inspired by "https://github.com/jeonsworld/ViT pytorch/blob/main/visualize_attention_map.ipynb"

MIT License

Copyright (c) 2020 jeonsworld

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import numpy as np
import cv2

def get_attention_map(image, attention, device, rotate=False):    
    image_size = image.shape[-1]

    att_mat = torch.stack(attention).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n].to(device), joint_attentions[n-1].to(device))

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    heatmap = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    heatmap = cv2.resize(heatmap / heatmap.max(), (image_size, image_size))[..., np.newaxis]
    # Duplicate heatmap to form a 3 channel image
    heatmap = np.concatenate([heatmap]*3, axis=2)

    image = image.permute(1, 2, 0).numpy()
    
    result = (heatmap * 2) + image
    
    if rotate:
        heatmap = np.rot90(heatmap)
        result = np.rot90(result)
        
    return result, heatmap