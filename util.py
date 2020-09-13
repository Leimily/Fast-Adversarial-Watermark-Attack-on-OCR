# coding:utf-8
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse


def invert(data):
    if data.max() < 1.5:
        return 1 - data
    else:
        return 255 - data


def transpose(data):  # trans 90 degree
    if len(data.shape) != 2:
        return np.swapaxes(data, 1, 2)
    else:
        return data.T


def cvt2raw(data):
    return transpose(invert(data))


def show(img):
    return cvt2Image(cvt2raw(img))


def clip(x, clip_min, clip_max):
    fix_min = x < clip_min
    x[fix_min] = clip_min
    fix_max = x > clip_max
    x[fix_max] = clip_max
    return x


def trans_back(X):
    return (X + 0.5) * 255


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args: sequences: a list of lists of type dtype where each element is a sequence
    Returns: A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    # 自动寻找序列的最大长度，形状为：batch_size * max_len
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape
    # return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


from trdg.generators import GeneratorFromStrings

# RGB格式颜色转换为16进制颜色格式
def RGB2Hex(RGB): # RGB is a 3-tuple
    color = '#'
    for num in RGB:
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def gen_wm(RGB):
    generator = GeneratorFromStrings(
        strings=['eccv'],
        count=1,  # 五种字体
        fonts=['fonts/Impact.ttf'],  # default: []
        language='en',
        size=100,  # 32
        skewing_angle=10,
        random_skew=False,
        blur=0,
        random_blur=False,
        # gaussian noise (0), plain white (1), quasicrystal (2) or picture (3)
        background_type=1,
        distorsion_type=0,  # None(0), Sine wave(1),Cosine wave(2),Random(3)
        distorsion_orientation=0,
        is_handwritten=False,
        width=-1,
        alignment=1,
        text_color=RGB2Hex(RGB),
        orientation=0,
        space_width=1.0,
        character_spacing=0,
        margins=(0, 0, 0, 0),
        fit=True,
    )
    img_list = [img for img, _ in generator]
    return img_list[0]


def get_text_mask(img: np.array):
    if img.max() <= 1:
        return img < 1 / 1.25
    else:
        return img < 255 / 1.25


def cvt2Image(array):
    if len(array.shape) == 3:
        array = array.reshape([array.shape[0], array.shape[1]])
    elif len(array.shape) == 4:
        array = array.reshape([array.shape[1], array.shape[2]])

    if array.max() <= 0.5:
        return Image.fromarray(((array + 0.5) * 255).astype('uint8'))
    elif array.max() <= 1:
        return Image.fromarray((array * 255).astype('uint8'))
    elif array.max() <= 255:
        return Image.fromarray(array.astype('uint8'))


def cvt2rgb(gray_img, text_mask):
    gray_img = invert(gray_img)
    op_mask = (~(gray_img == 1)) & (~text_mask)  # not_bg & not_text
    rgb_img = np.ones(list(gray_img.shape) + [3])
    rgb_img[:, :, :, 0] = gray_img
    rgb_img[:, :, :, 1] = gray_img
    rgb_img[:, :, :, 2] = gray_img
    rgb_img[op_mask, 0] = 1
    rgb_img[op_mask, 1] = (gray_img[op_mask] - 0.299) / 0.587
    rgb_img[op_mask, 2] = 0
    return invert(rgb_img)
