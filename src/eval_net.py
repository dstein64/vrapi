import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import torch

from model import Net
from utils import cifar10_loader, get_devices


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument(
        '--num-saved-images', type=int, default=100, help='Number of images to save per class.')
    devices = get_devices()
    parser.add_argument('--device', default='cuda' if 'cuda' in devices else 'cpu', choices=devices)
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    outdir = os.path.join(args.workspace, 'eval')
    os.makedirs(outdir, exist_ok=True)
    net = Net().to(args.device)
    net_path = os.path.join(args.workspace, 'net.pt')
    net.load_state_dict(torch.load(net_path, map_location='cpu'))
    test_loader = cifar10_loader(args.batch_size, train=False, shuffle=False)
    classes = test_loader.dataset.classes
    net.eval()
    saved_img_counts = [0] * 10
    y = []
    y_pred = []
    y_pred_proba = []
    y_repr = []
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
        y.extend(y_batch.tolist())
        outputs, representations = net(x_batch, include_penultimate=True)
        outputs = outputs.detach().cpu().numpy()
        representations = representations.detach().cpu().numpy()
        y_pred.extend(outputs.argmax(axis=1))
        y_pred_proba.extend(softmax(outputs, axis=1).tolist())
        y_repr.extend(representations.tolist())
        # Save example images.
        for idx, class_ in enumerate(y_batch.tolist()):
            if saved_img_counts[class_] >= args.num_saved_images:
                continue
            img_dir = os.path.join(outdir, 'images', f'{class_}_{classes[class_]}')
            os.makedirs(img_dir, exist_ok=True)
            img_arr = (x_batch[idx].detach().cpu().numpy() * 255).round().astype(np.uint8).transpose([1, 2, 0])
            img = Image.fromarray(img_arr)
            img_id = test_loader.batch_size * batch_idx + idx
            img.save(os.path.join(img_dir, f'{img_id}.png'))
            saved_img_counts[class_] += 1
    y = np.array(y)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    correct = y_pred == y
    np.savetxt(os.path.join(outdir, 'ground_truth.csv'), y, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(outdir, 'pred.csv'), y_pred, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(outdir, 'pred_proba.csv'), y_pred_proba, delimiter=',', fmt='%f')
    np.savetxt(os.path.join(outdir, 'correct.csv'), correct, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(outdir, 'representations.csv'), y_repr, delimiter=',', fmt='%f')
    cm = confusion_matrix(y, y_pred)
    print('Confusion Matrix:')
    print(cm)
    np.savetxt(os.path.join(outdir, 'confusion.csv'), cm, delimiter=',', fmt='%d')
    num_correct = correct.sum()
    total = len(y_pred)
    accuracy = num_correct / total
    eval_dict = {
        'correct': [num_correct],
        'total': [total],
        'accuracy': [accuracy]
    }
    eval_df = pd.DataFrame.from_dict(eval_dict)
    print('Evaluation:')
    print(eval_df)
    eval_df.to_csv(os.path.join(outdir, 'eval.csv'))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
