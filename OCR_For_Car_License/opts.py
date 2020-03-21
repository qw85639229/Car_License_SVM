import argparse

parser = argparse.ArgumentParser(description='ocr project')

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--lr', type=float, default=1e-2)

parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)

parser.add_argument('--start_epoch', type=int, default=0)

parser.add_argument('--epochs', type=int, default=30000)

parser.add_argument('--lr_steps', default=[20000], type=float, nargs="+")

parser.add_argument('--gpus', nargs='+', type=int, default=None)

parser.add_argument('--print-freq', '-p', default=5, type=int)

parser.add_argument('--eval-freq', '-ef', default=200, type=int)

parser.add_argument('--num_per_class', type=int, default=8)
