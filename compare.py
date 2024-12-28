import torch
import os
from PIL import Image
from torchvision import transforms
import skimage.color as sc
import numpy as np
import cv2

def psnr(gt_image, target_image):
    '''
    PSNR 계산.
    GT이미지와 대상 이미지가 동일 할 경우 inf 반환(PSNR이 정의되지 않음).
    '''
    return 10.0 * torch.log10(1.0 / torch.mean((gt_image - target_image) ** 2))

def get_avg_psnr(images, rgb2gray_f=None, scale_factor=3):
    '''
    이미지 데이터와 grayscale 변환 함수를 넣으면 scale_factor에 따라 평균 psnr을 구하는 함수.
    rgb2gray_f는 RGB PIL Image를 입력으로 받고 Grayscale PIL Image를 리턴해야 한다.
    '''
    # GT이미지와 LR이미지 획득(scale_factor에 따라 Bicubic하게 변환)
    n = len(images)
    GTs = [image.copy() for image in images]
    LRs = []
    for gt in GTs:
        w, h = gt.size
        lr = gt.resize((w//scale_factor, h//scale_factor)).resize((w,h), Image.BICUBIC)
        LRs.append(lr)

    # 주어진 rgb2gray 함수로 각 이미지 변환
    if rgb2gray_f:
        for i in range(n):
            gt, lr = GTs[i], LRs[i]
            GTs[i] = rgb2gray_f(gt)
            LRs[i] = rgb2gray_f(lr)

    # 각각 psnr 계산해서 평균 psnr 계산한 뒤 return
    psnr_sum = 0
    for gt, lr in zip(GTs, LRs):
        psnr_sum += psnr(transforms.ToTensor()(gt), transforms.ToTensor()(lr))
    
    return psnr_sum / n

def PIL_r2g(image):
    '''PIL convert를 이용해 rgb이미지를 grayscale로 바꾸는 함수'''
    g = image.copy()
    g = g.convert('YCbCr')
    g = transforms.ToTensor()(g)[0].unsqueeze(0)
    g = transforms.ToPILImage()(g)
    return g

def skimage_r2g(image):
    '''skimage.color rgb2ycbcr을 이용해 rgb이미지를 grayscale로 바꾸는 함수'''
    g = image.copy()
    g = sc.rgb2ycbcr(np.float32(g) / 255)[:, :, 0]
    g = transforms.ToTensor()(g) / 255
    g = transforms.ToPILImage()(g)
    return g

def opencv_r2g_YCrCb(image):
    '''opencv cvtColor을 이용해 rgb이미지를 grayscale로 바꾸는 함수(YCrCb)'''
    g = image.copy()
    rgb_narray = np.array(g)
    g = cv2.cvtColor(rgb_narray, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    g = transforms.ToPILImage()(g)
    return g

def opencv_r2g_YUV(image):
    '''opencv cvtColor을 이용해 rgb이미지를 grayscale로 바꾸는 함수(YUV)'''
    g = image.copy()
    rgb_narray = np.array(image)
    g = cv2.cvtColor(rgb_narray, cv2.COLOR_RGB2YUV)[:, :, 0]
    g = transforms.ToPILImage()(g)
    return g

def rgb_to_ycbcr_bt601(pil_img, return_y_channel=True):
    """
    PIL RGB 이미지를 BT.601 표준을 사용하여 YCbCr로 변환 후 PIL 이미지로 반환.
    Args:
        pil_img (PIL.Image.Image): RGB 형식의 PIL 이미지.
        return_y_channel (bool): True일 경우 Y 채널만 반환.
    Returns:
        PIL.Image.Image: YCbCr 또는 Y 채널 형식의 PIL 이미지.
    """
    # PIL 이미지를 NumPy 배열로 변환
    rgb_img = np.array(pil_img)

    # BT.601 변환 행렬
    matrix_bt601 = np.array([[0.299, 0.587, 0.114],
                            [-0.168736, -0.331264, 0.5],
                            [0.5, -0.418688, -0.081312]])

    # 변환 수행
    ycbcr_img = np.dot(rgb_img, matrix_bt601.T) + [0, 128, 128]

    if return_y_channel:
        # Y 채널만 추출
        y_channel = ycbcr_img[:, :, 0]
        return Image.fromarray(np.uint8(y_channel), mode='L')  # L 모드는 그레이스케일 이미지

    # NumPy 배열을 PIL 이미지로 변환 (YCbCr 모드로 설정)
    ycbcr_pil_img = Image.fromarray(np.uint8(ycbcr_img), mode='YCbCr')

    return ycbcr_pil_img


def rgb_to_ycbcr_bt709(pil_img, return_y_channel=True):
    """
    PIL RGB 이미지를 BT.709 표준을 사용하여 YCbCr로 변환 후 PIL 이미지로 반환.
    Args:
        pil_img (PIL.Image.Image): RGB 형식의 PIL 이미지.
        return_y_channel (bool): True일 경우 Y 채널만 반환.
    Returns:
        PIL.Image.Image: YCbCr 또는 Y 채널 형식의 PIL 이미지.
    """
    # PIL 이미지를 NumPy 배열로 변환
    rgb_img = np.array(pil_img)

    # BT.709 변환 행렬
    matrix_bt709 = np.array([[0.2126, 0.7152, 0.0722],
                            [-0.114572, -0.385428, 0.5],
                            [0.5, -0.454153, -0.045847]])

    # 변환 수행
    ycbcr_img = np.dot(rgb_img, matrix_bt709.T) + [0, 128, 128]

    if return_y_channel:
        # Y 채널만 추출
        y_channel = ycbcr_img[:, :, 0]
        return Image.fromarray(np.uint8(y_channel), mode='L')  # L 모드는 그레이스케일 이미지

    # NumPy 배열을 PIL 이미지로 변환 (YCbCr 모드로 설정)
    ycbcr_pil_img = Image.fromarray(np.uint8(ycbcr_img), mode='YCbCr')

    return ycbcr_pil_img

if __name__ == '__main__':
    img_dir = 'validation-images/' # your path to validation datasets
    sf = 3 # scale-factor

    img_names = [name for name in os.listdir(img_dir) if name.lower().endswith((".jpg", ".jpeg", ".png"))]
    img_paths = [os.path.join(img_dir, name) for name in img_names]
    images = []

    for path in img_paths:
        with Image.open(path) as img:
            images.append(img.convert('RGB'))

    print("Scale Factor: {}".format(sf))
    print("RGB psnr:                    {}".format(get_avg_psnr(images, scale_factor=sf)))
    print("PIL grayscale psnr:          {}".format(get_avg_psnr(images, PIL_r2g, scale_factor=sf)))
    print("skimage ycbcr psnr:          {}".format(get_avg_psnr(images, skimage_r2g, scale_factor=sf)))
    print("opencv YCrCb grayscale psnr: {}".format(get_avg_psnr(images, opencv_r2g_YCrCb, scale_factor=sf)))
    print("opencv YUV grayscale psnr:   {}".format(get_avg_psnr(images, opencv_r2g_YUV, scale_factor=sf)))
    print("BT.601 grayscale psnr:       {}".format(get_avg_psnr(images, rgb_to_ycbcr_bt601, scale_factor=sf)))
    print("BT.709 grayscale psnr:       {}".format(get_avg_psnr(images, rgb_to_ycbcr_bt709, scale_factor=sf)))