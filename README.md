# AnchorDETR

This is a AnchorDETR implementation based on [paddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

## Start
1. clone this repo and install requirements
    ```bash
    git clone https://github.com/Rapisurazurite/AnchorDETR-paddle.git
    cd AnchorDETR-paddle
    pip install -r requirements.txt
    ```
2. install paddleDetection
    ```bash
    python setup.py install
    ```
3. place your dataset in `dataset` folder
4. train
    ```bash
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c ./configs/anchor_detr/anchor_detr_r50_1x_coco_dc.yml --eval --fleet --acum_steps=2 -o LearningRate.base_lr=0.0001 log_iter=100
    ```
5. eval
    ```bash
    python tools/eval.py -c ./configs/anchor_detr/anchor_detr_r50_1x_coco_dc.yml
    ```
# Citatoins
```bibtex
@misc{wang2021anchor,
      title={Anchor DETR: Query Design for Transformer-Based Detector},
      author={Yingming Wang and Xiangyu Zhang and Tong Yang and Jian Sun},
      year={2021},
      eprint={2109.07107},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

