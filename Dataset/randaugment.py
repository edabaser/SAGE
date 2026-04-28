"""
Medikal Görüntüler (Dermoskopi) İçin Düzenlenmiş RandAugment

Orijinal FixMatch augmentation havuzundan kaldırılanlar:
  - Color     : Renk doygunluğunu değiştiriyor → melanin tespitini bozar
  - Equalize  : Histogram eşitleme → leke rengini değiştirir
  - Posterize : Renk kanallarını kırpar → ince doku bilgisini yok eder
  - Solarize  : Piksel değerlerini tersine çevirir → tamamen bozar
  - SolarizeAdd: Aynı sebep
  - Invert    : Tüm renkleri tersine çevirir → tanısal anlam kaybolur

Korunanlar (geometrik + hafif kontrast):
  AutoContrast, Brightness (düşük şiddet), Contrast (düşük şiddet),
  Identity, Rotate, Sharpness, ShearX, ShearY, TranslateX, TranslateY

Cutout düzeltmesi:
  Orijinal kod: CutoutAbs(img, int(32*0.5)) = 16px → 224px için anlamsız küçük
  Düzeltilmiş : CutoutAbs(img, 56) = %25 oranında kesme → medikal açıdan makul
"""

import logging
import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def CutoutAbs(img, v, **kwarg):
    """Görüntünün v×v piksellik bölgesini gri ile doldurur."""
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    color = (127, 127, 127)  # nötr gri
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Identity(img, **kwarg):
    return img


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    """
    Medikal dermoskopi görüntüleri için güvenli augmentasyon havuzu.
    Renk bozucu işlemler (Color, Equalize, Posterize, Solarize, Invert) ÇIKARILDI.
    Brightness ve Contrast şiddeti düşürüldü (0.9 → 0.4, bias küçültüldü).
    """
    augs = [
        (AutoContrast, None, None),
        (Brightness,   0.4,  0.05),   # Hafif parlaklık değişimi
        (Contrast,     0.4,  0.05),   # Hafif kontrast değişimi
        (Identity,     None, None),
        (Rotate,       30,   0),      # ±30 derece dönüş — dermoskopide geçerli
        (Sharpness,    0.6,  0.05),   # Hafif keskinleştirme
        (ShearX,       0.2,  0),      # Düşük kesme (0.3 → 0.2)
        (ShearY,       0.2,  0),
        (TranslateX,   0.2,  0),      # Küçük öteleme (0.3 → 0.2)
        (TranslateY,   0.2,  0),
    ]
    return augs


class RandAugmentMC(object):
    """
    FixMatch'te kullanılan RandAugment.

    n: kaç augmentasyon uygulanacak (2 önerilir)
    m: augmentasyon şiddeti (1-10 arası; medikal için 7 önerilir)

    Cutout boyutu:
      - Orijinal (CIFAR-32px): CutoutAbs(img, 16)   → görüntünün %50'si
      - Düzeltilmiş (HAM-224px): CutoutAbs(img, 56) → görüntünün %25'i
    """

    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)

        # Cutout: görüntü boyutuna göre dinamik
        w, h = img.size
        cutout_size = int(min(w, h) * 0.25)  # %25 oranı — sabit 16px değil
        img = CutoutAbs(img, cutout_size)
        return img
