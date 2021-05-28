import argparse
import json
import os
import sys

import torch

from model import Net
from utils import cifar10_loader, get_devices, set_seed


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workspace', type=str, default='workspace')
    devices = get_devices()
    parser.add_argument('--device', default='cuda' if 'cuda' in devices else 'cpu', choices=devices)
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    set_seed(args.seed)
    net = Net().to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    train_loader = cifar10_loader(args.batch_size, train=True)
    with open(os.path.join(args.workspace, 'classes.json'), 'w') as f:
        json.dump(train_loader.dataset.classes, f, indent=2)
    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'epoch: {epoch}/{args.epochs}, train loss: {train_loss:.3f}')
    net_path = os.path.join(args.workspace, 'net.pt')
    torch.save(net.state_dict(), net_path)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
