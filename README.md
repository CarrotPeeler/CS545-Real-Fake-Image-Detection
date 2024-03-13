# CS545-Real-Fake-Image-Detection

## Dependencies
Install the required packages by running the following:
```bash
pip install -r requirements.txt
```

## Getting the Data
We use a subset of the [Sentry dataset](https://huggingface.co/datasets/InfImagine/FakeImageDataset) for training and testing models. All images are compressed as JPGs to reduce size and mimic the format used most commonly for uploading images to websites and the internet. 

Run the following script to download, compress, and generate the subset of images and metadata used from Sentry.
```bash
cd CS545-Real-Fake-Image-Detection
python3 make_sentry_subset.py < /dev/null > log.txt 2>&1 &
```
