
# Y-channel PSNR Comparison
SR 논문에서 자주 사용되는 PSNR 측정 방법인 ground-truth image와 target image를 YCbCr color space로 변환 후 Y-channel만 취해 비교하는 방법은 python에서 다양한 라이브러리의 다양한 함수로 구현될 수 있다. 각각의 방식이 서로 다른 결과 값을 보이기에 각각을 비교해본다.

## 실험된 라이브러리/함수
- PIL - convert('YCbCr')
- Scikit Image - rgb2ycbcr()
- OpenCV - cvtColor(..., cv2.COLOR_RGB2YCrCb)
- OpenCV - cvtColor(..., cv2.COLOR_RGB2YUV)
- ITU-R 지정 BT.601 표준을 구현한 RGB to YUV 함수
- ITU-R 지정 BT.709 표준을 구현한 RGB to YUV 함수

## 비교를 위한 고정값
- PSNR 계산 함수: 직접 구현
- Bicubic Interpolation 알고리즘: PIL이미지 resize함수 사용(PIL.Image.BICUBIC)
- Scale factor: 3 (변경 가능)
- Validation Images: Set 5 (변경 가능)

## 실험 결과
```
Scale Factor: 3
===== Resizing with PIL.Image.resize() =====
RGB psnr:                    28.606664657592773
PIL grayscale psnr:          29.061071395874023
skimage ycbcr psnr:          30.37847328186035
opencv YCrCb grayscale psnr: 29.0618953704834
opencv YUV grayscale psnr:   29.0618953704834
BT.601 grayscale psnr:       29.061508178710938
BT.709 grayscale psnr:       29.07138442993164
===== Resizing with MATLAB style imresize() =====
RGB psnr:                    25.947063446044922
PIL grayscale psnr:          27.098241806030273
skimage ycbcr psnr:          28.418691635131836
opencv YCrCb grayscale psnr: 27.098852157592773
opencv YUV grayscale psnr:   27.098852157592773
BT.601 grayscale psnr:       27.09897232055664
BT.709 grayscale psnr:       26.946773529052734
```

## Run
1. `validation-images/` 폴더 밑에 validation에 사용할 이미지 데이터를 추가
2. `compare.py` 실행