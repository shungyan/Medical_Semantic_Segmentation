import argparse
import numpy as np
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

def main():
    # Load the TFLite model
    interpreter = make_interpreter("test_quant.tflite", device=':0')

if __name__ == '__main__':
  main()