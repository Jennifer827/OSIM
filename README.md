# OSIM

## Install

```
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install -v -e .
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
```

## Evaluation

```
python3 tools/OSIM.py image -n yolox-x -c ./yolox_x.pth --ref_path [ref image or directory] --test_path [test image or directory] --model_name [reconstruction model name] --dataset [dataset name] --data [data name] --conf 0.50 --nms 0.45 --tsize 640 --save_result --device gpu
```

## Acknowledgement

In this study, we utilized the YOLOX code. We sincerely thank the developers for sharing the code.

https://github.com/Megvii-BaseDetection/YOLOX

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
