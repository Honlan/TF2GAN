# UNet

Convolutional Networks for Biomedical Image Segmentation

[paper](https://arxiv.org/abs/1505.04597)

![](asset/teaser.png)

![](asset/result.jpg)

## Train

Make a folder under `dataset` and put your images as well as labels in it, just like the `CelebAMask19` folder

Convert the data to tfrecord for convenience, where the default value of `--dataset_name` is `CelebAMask19`

```
python main.py --dataset_name your_dataset_name --phase tfrecord
```

Train the model

```
python main.py --dataset_name your_dataset_name --phase train
```

## Test

Test the model. You need to specify the test folder by `--test_img_dir`, where the default value is `img`

```
python main.py --dataset_name your_dataset_name --phase test --test_img_dir your_test_img_dir
```

Or download the pretrained model of `CelebAMask19`, unzip it and you will get a folder named `output`

```
python main.py --phase test
```

The segmentation results are saved in `output/UNet_{dataset_name}/result/{test_img_dir}`