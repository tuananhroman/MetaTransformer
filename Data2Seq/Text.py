import torch
from torch import nn
import open_clip


def get_text_embeddings(text, tar_dim=768):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, t1, t2 = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    text_tensor = open_clip.get_tokenizer("ViT-B-32")(text)
    encoding = model.encode_text(text_tensor)
    encoding = zero_padding(encoding, tar_dim, device)
    return encoding


def zero_padding(text_tensor: torch.Tensor, tar_dim, device=None):
    padding_size = tar_dim - text_tensor.shape[1]
    zero_tensor = torch.zeros((text_tensor.shape[0], padding_size), device=device)
    padded_tensor = torch.cat([text_tensor.to(device), zero_tensor], dim=1)
    return padded_tensor
