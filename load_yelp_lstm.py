import argparse
from pathlib import Path

import torch

import dataloader
import train_classifier
from train_classifier import Model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a saved Yelp LSTM checkpoint and print diagnostics."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"D:\Users\22004\Downloads\yelp"),
        help="Path to the saved PyTorch checkpoint/state_dict.",
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path(r"E:\vsmodel\limeattack\data\embedding\glove.6B.200d.txt"),
        help="Path to the embedding file used when the model was trained.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the model onto, such as 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="this restaurant was surprisingly good and the service was excellent",
        help="Optional text to run through the model after loading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataloader.device = device
    train_classifier.device = device

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state_dict-like object, got {type(checkpoint)!r}")

    emb_weight = checkpoint["emb_layer.embedding.weight"]
    out_weight = checkpoint["out.weight"]
    hidden_size = out_weight.shape[1]
    nclasses = out_weight.shape[0]
    embedding_dim = emb_weight.shape[1]

    model = Model(
        embedding=str(args.embedding),
        hidden_size=hidden_size,
        depth=1,
        dropout=0.3,
        cnn=False,
        nclasses=nclasses,
    ).to(device)

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Embedding file: {args.embedding}")
    print(f"Device: {device}")
    print(f"Classes: {nclasses}")
    print(f"Hidden size: {hidden_size}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Vocabulary size from checkpoint: {emb_weight.shape[0]}")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    if args.text:
        probs = model.text_pred([args.text.split()]).detach().cpu()[0]
        pred = int(torch.argmax(probs).item())
        print(f"Text: {args.text}")
        print(f"Probabilities: {probs.tolist()}")
        print(f"Predicted class: {pred}")


if __name__ == "__main__":
    main()
