###  MABDT: Multi-scale Attention Boosted Deformable Transformer for Remote Sensing Image Dehazing

**Abstract**: Owing to the heterogeneous spatial distribution and non-uniform morphological characteristics of haze in remote sensing images (RSIs), conventional dehazing algorithms struggle to precisely recover the fine-grained details of terrestrial objects. To address this issue, a novel multi-scale attention boosted deformable Transformer (MABDT) tailored for RSI dehazing was proposed. This framework synergizes the multi-receptive field features elicited by convolutional neural network (CNN) with the long-term dependency features derived from Transformer, which facilitates a more adept restitution of texture and intricate detail information within RSIs. Firstly, spatial attention deformable convolution was introduced for computation of multi-head self-attention in the Transformer block, particularly in addressing complex haze scenarios encountered in RSIs. Subsequently, a multi-scale attention feature enhancement (MAFE) block was designed, tailored to capture local and multi-level detailed information features using multi-receptive field convolution operations, thereby accommodating non-uniform haze. Finally, a multi-level feature complementary fusion (MFCF) block was proposed, leveraging both shallow and deep features acquired from all encoding layers to augment each level of reconstructed image. The dehazing performance was evaluated on five open-source datasets, and quantitative and qualitative experimental results demonstrate the advancements of the proposed method in both metrical scores and visual quality. 

### Requirements
```python
python 3.9.18
pytorch 2.1.2
torchvision 0.16.2
pillow 10.0.1
scikit-image 0.22.0
timm 0.9.12
tqdm 4.66.1
opencv-python 4.8.1.78
```
### Pre-trained models
Kindly access the pre-trained models of our MABDT through the provided links for download.

| Dataset           | Link                                                         | Dataset              | Link                                                         |
| ----------------- | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| StateHaze1K-thick | [[Baidu Cloud](https://pan.baidu.com/s/1UEQgzQY0mFUIbyjpSAZdgg), code: 7fkg] | StateHaze1K-moderate | [[Baidu Cloud](https://pan.baidu.com/s/1EE9zRgTdmtUCCvBKVNbIow), code: dvqj] |
| StateHaze1K-thin  | [[Baidu Cloud](), code: ]                                    | RICE1                | [[Baidu Cloud](https://pan.baidu.com/s/1XrgP-h3FIpomUSjNgt9oeQ), code: 7sik] |
| RICE2             | [[Baidu Cloud](https://pan.baidu.com/s/1lRg6dO1LK5277R0gltWSmg), code: 94h8] |                      |                                                              |

### Train
For those who wish to perform rapid training on their custom datasets, we provide a straightforward training code in the `train.py` file, enabling training of our MABDT. Please refer to the example usage within the file to train the model on your datasets.
```python
python train.py --seed 2023 --epoch 100 --batch_size_train 1 --batch_size_val 1 --patch_size_train 256 --patch_size_val 256 --lr 2e-4 --lr_min 1e-8 --train_data ./StateHaze1K-thick/hazy --val_data ./StateHaze1K-thick/clear --resume_state ./model_latest.pth --save_state ./model_best.pth --cuda True --val_frequency 1 --lp_weight 0.05 --lg_weight 0.08 --only_last False --autocast True --num_works 1
```
### Test
To evaluate our EMPF-Net on your own datasets or publicly available datasets, you can utilize the following example usage to conduct experiments.

```python
python test.py --val_data ./Haze1k_thick/hazy --result_dir ./Haze1k_thick/test/result/ --resume_state ./models/Haze1K-thick.pth --expand_factor 128 --result_save True --cuda True --only_last True --num_works 1
```
### Citation

If you find our work helpful for your research, please consider citing our work following this.

```python

```

### Contact  us

If I have any inquiries or questions regarding our work, please feel free to contact us at [ningjin@cdut.edu.cn](ningjin@cdut.edu.cn).