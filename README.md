# CS545-Real-Fake-Image-Detection
Repository for developing Fake-Real Image Detection models using the Active Learning framework.

## Dependencies
Install the required packages by running the following:
```bash
pip install -r requirements.txt
```

## Getting the Data
We use a subset of the [Sentry dataset](https://huggingface.co/datasets/InfImagine/FakeImageDataset) for training and testing models. All images are compressed as JPGs to reduce size and mimic the format used most commonly for uploading images to websites and the internet. The subset we partition contains 240k fake and 240k real images.

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

## Training the Models


## Performing Inference