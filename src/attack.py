import argparse
import os
import sys
import time

from cleverhans.torch.attacks import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import torch

from model import Net
from utils import ATTACKS, cifar10_loader, get_devices, set_seed


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument(
        '--num-saved-images', type=int, default=100, help='Number of images to save per class for each attack.')
    devices = get_devices()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if 'cuda' in devices else 'cpu', choices=devices)
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    set_seed(args.seed)
    net = Net().to(args.device)
    net_path = os.path.join(args.workspace, 'net.pt')
    net.load_state_dict(torch.load(net_path, map_location='cpu'))
    test_loader = cifar10_loader(args.batch_size, train=False, shuffle=False)
    classes = test_loader.dataset.classes
    eval_correct = np.loadtxt(os.path.join(args.workspace, 'eval', 'correct.csv'), dtype=bool, delimiter=',')
    net.eval()
    for attack in ATTACKS:
        outdir = os.path.join(args.workspace, 'attack', attack)
        os.makedirs(outdir, exist_ok=True)
        saved_img_ids = list(reversed(np.where(eval_correct)[0]))
        saved_img_counts = [0] * 10
        y = []
        y_pred = []
        y_pred_proba = []
        y_repr = []
        norms = {'l0': [], 'l1': [], 'l2': [], 'linf': []}
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            # Limit attack to images that were correctly classified initially.
            offset = test_loader.batch_size * batch_idx
            correct_idxs = np.where(eval_correct[offset:offset + len(x_batch)])[0]
            if correct_idxs.size == 0:
                continue
            x_batch, y_batch = x_batch[correct_idxs], y_batch[correct_idxs]
            # Log time.
            print(attack, len(y), time.time())
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            if attack == 'fgsm':
                # Modify each pixel by up to 3 intensity values.
                x_adv_batch = fast_gradient_method(net, x_batch, 3 / 255, float('inf'))
            elif attack == 'bim':
                # Modify each pixel by up to 1 intensity value per iteration, for 10 iterations.
                # Clip to 3 intensity values.
                x_adv_batch = projected_gradient_descent(
                    net, x_batch, 3 / 255, 1 / 255, 10, float('inf'), rand_init=False)
            elif attack == 'cw':
                x_adv_batch = carlini_wagner_l2(net, x_batch, 10)
            else:
                raise RuntimeError('Unsupported attack: ' + attack)
            if attack != 'cw':
                # Match the quantization of the non-adversarial images. Don't use this with C&W, as
                # it removes the attack's effectiveness (quantizing essentially reverts the attack
                # since it's an l2-based attack that results in small per-pixel perturbations).
                x_adv_batch = ((x_adv_batch * 255).round() / 255.0)
            x_adv_batch = x_adv_batch.clip(0, 1)
            perturb_batch = (x_batch - x_adv_batch).flatten(start_dim=1)
            for p in [0, 1, 2, float('inf')]:
                norms[f'l{p}'].extend(perturb_batch.norm(p=p, dim=1).tolist())
            y.extend(y_batch.tolist())
            outputs, representations = net(x_adv_batch, include_penultimate=True)
            outputs = outputs.detach().cpu().numpy()
            representations = representations.detach().cpu().numpy()
            y_pred.extend(outputs.argmax(axis=1))
            y_pred_proba.extend(softmax(outputs, axis=1).tolist())
            y_repr.extend(representations.tolist())
            # Save example perturbed images.
            for idx, class_ in enumerate(y_batch.tolist()):
                if saved_img_counts[class_] >= args.num_saved_images:
                    continue
                img_dir = os.path.join(outdir, 'images', f'{class_}_{classes[class_]}')
                os.makedirs(img_dir, exist_ok=True)
                img_arr = (x_adv_batch[idx].detach().cpu().numpy() * 255).round().astype(np.uint8).transpose([1, 2, 0])
                img = Image.fromarray(img_arr)
                img.save(os.path.join(img_dir, f'{saved_img_ids.pop()}.png'))
                saved_img_counts[class_] += 1
        y = np.array(y)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        correct = y_pred == y
        y_repr = np.array(y_repr)
        np.savetxt(os.path.join(outdir, 'pred.csv'), y_pred, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(outdir, 'pred_proba.csv'), y_pred_proba, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(outdir, 'correct.csv'), correct, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(outdir, 'representations.csv'), y_repr, delimiter=',', fmt='%f')
        norms_df = pd.DataFrame.from_dict(norms)
        norms_df.to_csv(os.path.join(outdir, 'norms.csv'))
        norms_df.describe().to_csv(os.path.join(outdir, 'norms_stats.csv'))
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
