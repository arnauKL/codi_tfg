#!/usr/bin/env python3

import torch


def printversions():
    print("pytorch: ", torch.__version__)
    print("GPU Available: ", {torch.cuda.is_available()})


def main() -> None:
    printversions()
    print("done")


if __name__ == "__main__":
    main()
