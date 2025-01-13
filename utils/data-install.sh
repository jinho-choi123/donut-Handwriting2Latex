#!/bin/bash

DATA_DIR="data";
ZIPFILE_NAME="mathwriting.tgz";
ZIPFILE_PATH="$DATA_DIR/$ZIPFILE_NAME";

UNZIPDIR_NAME="mathwriting-2024";
UNZIPDIR_PATH="$DATA_DIR/$UNZIPDIR_NAME"

# make data directory if not exists
mkdir -p "$DATA_DIR"

# install math-writing dataset
if [ -e $ZIPFILE_PATH ]; then
	echo "$ZIPFILE_PATH - Already exists...";
else
	echo "$ZIPFILE_PATH - Doesn't exists. Starting download...";
	wget -O $ZIPFILE_PATH https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz;
fi

# check if unzip directory exists
if [ -d "$UNZIPDIR_PATH" ]; then 
	echo "$UNZIPDIR_PATH - Already exists...";
else 
	echo "$UNZIPDIR_PATH - Doesnt exists. Starting untar...";
	tar -xvzf "$ZIPFILE_PATH" -C "$DATA_DIR" --strip-components=1;
fi
