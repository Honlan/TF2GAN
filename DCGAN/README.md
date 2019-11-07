# DCGAN

Deep Convolutional Generative Adversarial Network

[arvix](https://arxiv.org/abs/1511.06434) [project](https://github.com/carpedm20/DCGAN-tensorflow)

![](DCGAN/asset/teaser.png)

![](DCGAN/asset/result.jpg)

## Train

Make a folder under `dataset` and put your images in it, just like the `celeba` dataset

Train the model

```
python main.py
```

## Test

Download the pretrained model, unzip it and you will get a folder named `output`

Test the model

```
python main.py --phase test
```

The generated image is saved as `output/DCGAN_{dataset_name}/result/result.jpg`