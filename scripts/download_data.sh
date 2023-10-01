#!/bin/bash

mkdir ../MData && cd ../MData

echo "Downloading Flickr8k text ..."
curl -O -L  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

echo "Downloading Flickr8k dataset ..."
curl -O -L https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

echo "Downloading Glove ..."
curl -O https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip

echo "Unzipping data ..."
unzip Flickr8k_Dataset.zip -d Flicker8k_Dataset
unzip Flickr8k_text.zip -d Flicker8k_text
unzip glove.6B.zip

echo "keeping only glove.6B.50d.txt ..."
rm glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt

rm Flickr8k_Dataset.zip Flickr8k_text.zip glove.6B.zip
