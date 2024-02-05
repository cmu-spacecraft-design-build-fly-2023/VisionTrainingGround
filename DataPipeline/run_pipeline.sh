#!/bin/bash

# Default values for configurations
BOUNDS="-84 24 -78 32"
IDATE="2020-05-01"
FDATE="2023-12-31"
LANDSAT=8
MAXIMS=50
SCALE=150
BASE_OUTPATH="Landsat_Data"
FINAL_OUTPUT_PATH="Landsat_Data/17R_dataset"

# Read command line arguments for configuration
while getopts ":b:i:f:l:m:s:o:y:" opt; do
  case $opt in
    b) BOUNDS="$OPTARG"
    ;;
    i) IDATE="$OPTARG"
    ;;
    f) FDATE="$OPTARG"
    ;;
    l) LANDSAT="$OPTARG"
    ;;
    m) MAXIMS="$OPTARG"
    ;;
    s) SCALE="$OPTARG"
    ;;
    o) BASE_OUTPATH="$OPTARG"
    ;;
    y) FINAL_OUTPUT_PATH="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Array of keys to iterate through
KEYS=("17R") # Add or remove keys as needed

# Main processing loop
for KEY in "${KEYS[@]}"; do
  # Run earthenginedl.py
  python3 earthenginedl.py --bounds $BOUNDS --idate "$IDATE" --fdate "$FDATE" --landsat $LANDSAT --region "$KEY" --maxims $MAXIMS --scale $SCALE --outpath "$BASE_OUTPATH/$KEY"
  
  # Run saliencymap.py
  python saliencymap.py --dir_path "$BASE_OUTPATH/$KEY" --crs EPSG:4326 --grid_key "$KEY"
  
  # Run saliencymap2boxes.py
  python saliencymap2boxes.py -k "$KEY" -w 100 -n 1000 -p "$BASE_OUTPATH/$KEY/landmarks"
  
  # Run prepare_yolo_data.py with configurable output path
  python prepare_yolo_data.py --data_path "$BASE_OUTPATH/$KEY" --output_path "$FINAL_OUTPUT_PATH" --r "$KEY"
done