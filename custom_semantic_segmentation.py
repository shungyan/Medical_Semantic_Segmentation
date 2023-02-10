import argparse
import numpy as np
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import segment
# from pycoral.utils.edgetpu import make_interpreter
import tflite_runtime.interpreter as tflite

def main():
    # Load the TFLite model
    # interpreter = make_interpreter("test_quant.tflite", device=':0')
    interpreter = tflite.Interpreter(model_path="model.tflite")

if __name__ == '__main__':
  main()