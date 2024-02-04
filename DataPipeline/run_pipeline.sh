#!/bin/bash

# Correct way to define the path to Python executable in Bash
python="C:\\Python311\\python.exe"  # Using forward slashes

# Configuration variables
DIR_PATH="/home/argus-vision/vision/VisionTrainingGround/DataPipeline/Landsat_Data"
CRS="EPSG:4326"
GRID_KEYS=("17R" "AnotherKey" "YetAnotherKey")  # Add or remove keys as needed
WINDOW_SIZE=100
NUM_BOXES=1000
LANDMARKS_PATH="${DIR_PATH}\\landmarks"  # Dynamically set based on DIR_PATH

# Iterate through each grid key and execute the scripts
for KEY in "${GRID_KEYS[@]}"
do
    echo "Processing for key: $KEY"

    # Use the Python variable to run the saliencymap.py script
    "$python" saliencymap.py --dir_path "$DIR_PATH" --crs "$CRS" --grid_key "$KEY"

    # Check for error in execution
    if [ $? -ne 0 ]; then
        echo "Error processing saliencymap.py for key $KEY."
        continue  # Skip to the next key
    fi

    # Use the Python variable to run the saliencymap2boxes.py script
    "$python" saliencymap2boxes.py -k "$KEY" -w "$WINDOW_SIZE" -n "$NUM_BOXES" -p "$LANDMARKS_PATH"

    # Check for error in execution
    if [ $? -ne 0 ]; then
        echo "Error processing saliencymap2boxes.py for key $KEY."
        continue  # Skip to the next key
    fi

    echo "Completed processing for key: $KEY"
done

echo "All processing completed."