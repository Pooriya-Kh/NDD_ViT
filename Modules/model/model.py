import torch
import torch.nn as nn
from transformers import ViTConfig, ViTImageProcessor, ViTForImageClassification
from copy import deepcopy
import os
from torchvision.transforms import Resize
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model.attention import get_attention_map

class ViT(nn.Module):
    def __init__(
        self,
        pretrained,
        model_name=None,
        device="cuda:0"
    ):
        super(ViT, self).__init__()
        
        self.pretrained = pretrained
        self.best_state = {
            "loss": {"value": 100, "state_dict": None},
            "acc": {"value": 0, "state_dict": None},
        }
        
        if self.pretrained:
            self.model_name = model_name
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=3,
                ignore_mismatched_sizes=True,
                attn_implementation="eager", #Read about this!
                hidden_dropout_prob=0.1
                # attention_probs_dropout_prob=0.1,
            ).to(device)
            
            self.image_processor = ViTImageProcessor.from_pretrained(
                model_name,
                do_resize=False,
                do_normalize=False,
                do_rescale=False
            )
            
        else:
            self.config = ViTConfig(
                image_size=image_size,
                patch_size=32,
                num_labels=3
            )
            self.vit = ViTForImageClassification(config).to(device)

            self.image_processor = ViTImageProcessor(
                do_resize=False,
                do_normalize=False,
                do_rescale=False
            )

        self.device = device

    def process_image(self, x):
        x = self.image_processor(images=x, return_tensors="pt")["pixel_values"]
        return x
        
    def forward(self, x):
        x = self.process_image(x).to(self.device)
        outputs = self.vit(x, output_attentions=True, output_hidden_states=True)
        return outputs.logits, outputs.attentions, outputs.hidden_states

    def infer(self,
              x,
              atlas_data,
              atlas_labels,
              show_overlaid_attention_map=True,
              show_patches=True,
              show_attention_map=True,
              show_input=True,
              return_att_map=False
             ):
        
        def min_max_scale(x):
            x = (x - x.min()) / (x.max() - x.min())
            return x

        def resize_original(x):
            resize = Resize(size=(570, 950), antialias=True)
            x = resize(x)
            return x

        def show_image(x, pred, region):
            fig, axes = plt.subplots(figsize=(10, 2.3),
                                     ncols=3,
                                     dpi=300,
                                     layout="tight")
            for i in range(3):
                im = axes[i].imshow(x[i, :, :], vmin=0, vmax=1)
                fig.colorbar(im, ax=axes[i])
                axes[i].axis("off")

            fig.suptitle(
                f"Prediction: {pred}\n Most Important Region: {region}",
                fontname="serif",
                fontsize="medium",
                fontweight="medium",
            )

            return fig, axes
            
        id2label = {0: 'CN', 1: 'MCI', 2: 'AD'}
        atlas_id2region = {value: key for key, value in atlas_labels.items()}

        logits, attentions, _ = self.forward(x)
        pred = id2label[logits.argmax(1).cpu().item()]
        overlaid_att_map, att_map = get_attention_map(x,
                                                      attentions,
                                                      self.device,
                                                      rotate=True)

        overlaid_att_map = torch.tensor(overlaid_att_map.copy()).permute(2, 0, 1)
        att_map = torch.tensor(att_map.copy()).permute(2, 0, 1)

        overlaid_att_map = resize_original(overlaid_att_map)
        att_map = resize_original(att_map)
        x = resize_original(x.rot90(k=1, dims=(1,2)))
        image_mask = torch.where(x>0, 1, 0)

        overlaid_att_map *= image_mask
        att_map *= image_mask

        x = min_max_scale(x)
        overlaid_att_map = min_max_scale(overlaid_att_map)
        att_map = min_max_scale(att_map)

        # Extracting the most important region
        atlas_mask = torch.where(atlas_data>0, 1, 0)
        atlas_masked_att_map = atlas_mask * att_map
        final = torch.where(atlas_masked_att_map==atlas_masked_att_map.max(), 1, 0)
        region = final * atlas_data
        region = int(region.max().item())
        region = atlas_id2region[region]        

        if show_overlaid_attention_map:
            fig, axes = show_image(overlaid_att_map, pred, region)
            
            # Drawing patches
            # why atlas_masked_att_map in original version?!!!
            # 0.98?!!!
            if show_patches:
                final = torch.where(overlaid_att_map>overlaid_att_map.max()*0.95, 1, 0)
                for i in range(3):
                    m = final[i, :, :].nonzero()
                    if m.numel() != 0:
                        for mm in m :
                            mm -= 10
                            rect = patches.Rectangle(
                                [mm[1], mm[0]],
                                20,
                                20,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='none'
                            )
                            axes[i].add_patch(rect)

            plt.savefig(
                f"{pred}_inference.pdf",
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.1
            )

        if show_attention_map:
            show_image(att_map, pred, region)

        if show_input:
            show_image(x, pred, region)

        if return_att_map:
            return pred, region, att_map
        else:
            return pred, region
        

    def save_best_state(self, metric, value):
        self.best_state[metric]["value"] = value
        self.best_state[metric]["state_dict"] = deepcopy(self.state_dict())

    def load_best_state(self, metric):
        self.load_state_dict(self.best_state[metric]["state_dict"])

    def save_best_state_file(self, metric, save_dir, file_name):
        file_name  = f"{file_name}_{metric}.pt"
        files = os.listdir(save_dir)
        
        if file_name in files:
            i = input("The file already exists. Do you want to replace it? (y/n)")
            if i == 'y':
                torch.save(self.best_state[metric]["state_dict"], save_dir + file_name)
                print("Model replaced!")
            else:
                print("Skipped!")
        
        else:
            torch.save(self.best_state[metric]["state_dict"], save_dir + file_name)
            print("Model saved!")

    def load_best_state_file(self, metric, load_dir, file_name):
        file_name  = f"{file_name}_{metric}.pt"
        self.load_state_dict(torch.load(load_dir + file_name))