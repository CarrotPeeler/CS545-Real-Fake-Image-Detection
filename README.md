# CS545-Real-Fake-Image-Detection
Repository for developing Fake-Real Image Detection models using the Active Learning framework.

## Dependencies
Install the required packages by running the following:
```bash
pip install -r requirements.txt
```

## Getting the Data
We use a subset of the [Sentry dataset](https://huggingface.co/datasets/InfImagine/FakeImageDataset) for training and testing models. All images are resized to 256x256 and compressed as JPGs to reduce size and mimic the format used most commonly for uploading images to websites and the internet. The subset we partition contains 240k fake and 240k real images.

Download the dataset from Kaggle [here](https://www.kaggle.com/datasets/carrotpeeler/sentry-subset). Alternatively, use the scripts below to manually download the fake and real data. The scripts allow for custom partition sizes of the fake data.

```bash
cd CS545-Real-Fake-Image-Detection
```
Run the following script to download, compress, and generate the subset of fake images and metadata used from Sentry.

Note: for each dataset in sentry, the respective tar files are downloaded together, extracted, and then deleted to save space. Meaning, you will at most need ~100 GBs to handle the intermediate downloading process. The final compressed subset is only ~18 GBs.
```bash
python3 datagen/make_sentry_subset.py < /dev/null > log.txt 2>&1 &
```
For real image data, we use CC3M (Google Conceptual Captions), FFHQ, and AFHQv2 for training, and CC3M and CelebA-HQ for testing.

For CC3M, we use 155k of the train images and all of the val images.

Run the following script to download all real data components:
```bash
python3 datagen/cc3m/add_real_data.py < /dev/null > log.txt 2>&1 &
```

After downloading the data, there should be 482k train images and 187k validation images.


## Active Learning
We leverage Active Learning in an attempt to improve/maintain model performance while reducing the overall size of the training data.

An example of how to use the Active Learning code is located in [UniversalFakeDetect/train.py](https://github.com/CarrotPeeler/CS545-Real-Fake-Image-Detection/blob/main/UniversalFakeDetect/train.py) under the `train_active_learning()` function.


## Available Models
The models currently available are listed below. The names listed can be directly input as an argument for the `--arch` option.
- Imagenet:resnet18
- Imagenet:resnet34
- Imagenet:resnet50
- Imagenet:resnet101
- Imagenet:resnet152
- Imagenet:vgg11
- Imagenet:vgg19
- Imagenet:vit_b_16
- CLIP:RN50
- CLIP:RN101
- CLIP:ViT-L/14


## Training the Models
Run the `train_normal.sh` script for normal training without Active Learning. Edit the arguments in the script to change the model and adjust training hyperparameters.
```bash
bash UniversalFakeDetect/train_normal.sh
```

To run training with Active Learning:
```bash
bash UniversalFakeDetect/train_active_learning.sh
```

Note: testing is automatically performed after Active Learning training finishes. The results dict will be printed to the output log file.



## Performing Inference
Run the following to perform inference for either normal or active learning checkpoints. Make sure to edit the path arguments for the checkpoint file and the save directory.
```bash
bash UniversalFakeDetect/test.sh
```


## LLaVA Real-Fake Explainer
### Training data
Explainer training data is generated with LLaVA 1.6-7b and source category-guided prompts as shown in `LLaVA_Experiments/llava_train_data.py`. The `real_fake_llava_train.json` file can be found [here](https://wpi0-my.sharepoint.com/:f:/g/personal/simran_wpi_edu/EjYNAq_1KQlImgOQlwbpZZQBvx7SfIxwpYqibTw3PBE90A?e=55azN8).

### Test Explanations
Human-like reasoning behind why our LoRA (PEFT) finetuned LLaVA model thinks a test image is real or fake can be found [here](https://wpi0-my.sharepoint.com/:f:/g/personal/simran_wpi_edu/EjYNAq_1KQlImgOQlwbpZZQBvx7SfIxwpYqibTw3PBE90A?e=55azN8) in the `test_explanations_and_classifications.txt` file.


## Acknowledgements
1. https://github.com/lunayht/DBALwithImgData
2. https://github.com/Yuheng-Li/UniversalFakeDetect
3. https://github.com/facebookresearch/SlowFast/blob/main/slowfast/utils/distributed.py
4. https://github.com/haotian-liu/LLaVA
