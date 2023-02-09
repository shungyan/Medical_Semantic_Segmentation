import argparse

import numpy as np
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
import tflite_runtime.interpreter as tflite

def main():
    interpreter = make_interpreter("model.tflite", device=':0')
    #interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    width, height = common.input_size(interpreter)
    print(width,height)

    img = Image.open('test.png')
    # resized_img, _ = common.set_resized_input(
    # interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
    # print(resized_img)
    print(type(img))
    print(img.size)
    resized_img = img.resize((width, height), Image.ANTIALIAS)
    print(type(resized_img))
    print(resized_img.size)
    resized_img = np.expand_dims(resized_img, axis=-1)
    print(type(resized_img))
    print(resized_img.shape)
    common.set_input(interpreter, resized_img)

    interpreter.invoke()

    result = segment.get_output(interpreter)
    print(result.shape)
    result = np.squeeze(result)
    print(result.shape)
    np.save("result.npy",result)

    # If keep_aspect_ratio, we need to remove the padding area.
    # new_width, new_height = (128,128)
    # result = result[:new_height, :new_width]
    mask_img = Image.fromarray(result)

    # Concat resized input image and processed segmentation results.
    # output_img = Image.new('RGB', (2 * new_width, new_height))
    # output_img.paste(resized_img, (0, 0))
    # output_img.paste(mask_img, (width, 0))
    mask_img.save('result.png')

if __name__ == '__main__':
  main()