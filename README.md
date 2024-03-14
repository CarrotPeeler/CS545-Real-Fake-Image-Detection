# CS545-Real-Fake-Image-Detection

## Dependencies
Install the required packages by running the following:
```bash
pip install -r requirements.txt
```

## Getting the Data
We use a subset of the [Sentry dataset](https://huggingface.co/datasets/InfImagine/FakeImageDataset) for training and testing models. All images are compressed as JPGs to reduce size and mimic the format used most commonly for uploading images to websites and the internet. 

Run the following script to download, compress, and generate the subset of fake images and metadata used from Sentry.
```bash
cd CS545-Real-Fake-Image-Detection
python3 datagen/make_sentry_subset.py < /dev/null > log.txt 2>&1 &
```
Note: for each dataset in sentry, the respective tar files are downloaded together, extracted, and then deleted to save space. Meaning, you will at most need ~100 GBs to handle the intermediate downloading process. The final compressed subset is only a few GBs. 

For real image data, we use CC3M (Google Conceptual Captions), CelebA-HQ, 
For CC3M, we use 200k of the train images and 8k of the val images.
