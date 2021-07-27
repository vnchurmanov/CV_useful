from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *


def imresize(image, size):
    """Функция, меняющая размер массива с помощью PIL"""
    pil_im = Image.fromarray(uint8(image))
    return array(pil_im.resize(image))


def equalize_hist(image, nbr_bins=256):
    """Выравнивание гистограммы полутонового изображения:
    нормировка яркости перед последующей обработкой,
    повышение контрасности"""
    # получаем гистограмму изображения
    imhist, bins = histogram(image.flatten(), nbr_bins)

    # cdf - cumulative distribution function -
    # функция распределения значений пикселей в изображении
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # нормируем

    # используем линейную интерполяцию cdf для нахождения
    # значений новых пикселей
    im2 = interp(image.flatten(), bins[:-1], cdf)
    return im2.reshape(image.shape), cdf

def compute_average(imlist):
    """Усредняющая функция, вычисляющая среднее списка изображений"""
    # открываем первое изображение и преобразуем в массив типа float
    average_im = array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            average_im += array(Image.open(imname))
        except:
            print(imname + '...пропущено')
            average_im /= len(imlist)

    # возвращаем среднее в виде массива значений типа uint8
    return array(average_im, 'uint8')

image = array(Image.open('car.jpeg').convert('L'))
im2, cdf = equalize_hist(image)
hist(im2.flatten(), 64)
show()
pil_image = Image.fromarray(uint8(im2))
imshow(pil_image)
show()