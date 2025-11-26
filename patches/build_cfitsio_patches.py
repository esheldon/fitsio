import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--moddir', required=True,
                        help='directory containing modified files')
    parser.add_argument('--dir', required=True,
                        help='directory containing unmodified files')
    parser.add_argument('--patch-dir', required=True)
    return parser.parse_args()


def main():
    args = get_args()

    os.makedirs(args.patch_dir, exist_ok=True)

    for root, _, files in os.walk(args.dir):
        for fname in files:
            src = os.path.join(args.dir, fname)
            dst = os.path.join(args.moddir, fname)
            patch = os.path.join(args.patch_dir, fname + '.patch')
            os.system('diff -u %s %s > %s' % (src, dst, patch))
            with open(patch, 'rb') as fp:
                buff = fp.read()
            if len(buff) == 0:
                os.remove(patch)
            else:
                print(fname)
        break


main()
