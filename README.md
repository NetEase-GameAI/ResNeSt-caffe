# ResNeSt-caffe

- A caffe version of official [PyTorch-ResNeSt](https://github.com/zhanghang1989/ResNeSt).

- Caffemodels are avaliable [here](https://drive.google.com/drive/folders/14MB8NTcQVYpEjxZ8So5cfBTmotBwWij-?usp=sharing)

### Model Results

| Model Name  | Crop Size | PyTorch Top1 | Caffe Top1 | Caffe Speed |
| :---------: | :-------: | :----------: | :--------: | :---------: |
| ResNeSt-50  |  224x224  |    81.03     |   81.11    |    6.3ms    |
| ResNeSt-101 |  256x256  |    82.83     |   83.06    |   13.1ms    |
| ResNeSt-200 |  320x320  |    83.84     |   84.22    |   39.5ms    |
| ResNeSt-269 |  416x416  |    84.54     |   84.67    |   88.2ms    |

### Convert Details

- We convert the official PyTorch-ResNeSt to Caffe by pipeline: PyTorch-ONNX-Caffe. 

- For exported ONNX model, we first merge Exp-ReduceSum-Div into one Softmax node. Then we convert to caffe by our onnx2caffe tools written from scratch.

- Caffe models are tested on single GTX-1080Ti. PyTorch results come from official PyTorch-ResNeSt.

- It seems caffe models are slower than that in [ResNeSt-paper](https://arxiv.org/abs/2004.08955)

  - Some ops may be more friendly for PyTorch, while less for Caffe.

  - We test on GTX-1080Ti while the latency in paper tested on Tesla-V100.

- Need [bvlc-caffe](https://github.com/BVLC/caffe) and Permute layer from [ssd-caffe](https://github.com/weiliu89/caffe/tree/ssd).

### Evaluation Tools

```python
python evaluation.py -imgs /data/ImageNet2012/val -label /data/ImageNet2012/labels/val.txt -proto resnest50.prototxt -model resnest50.caffemodel -size 224 -batch 20
```
