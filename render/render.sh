#!/bin/bash

if ! command -v convert &> /dev/null; then
    echo "ImageMagick is not installed. Installing..."
    sudo apt update && sudo apt install -y imagemagick
fi

# Convert all .ppm files to .png
for file in *.ppm; do
    if [ -f "$file" ]; then
        output_file="${file%.ppm}.png"
        echo "Converting $file to $output_file..."
        convert "$file" "$output_file"
    fi
done

echo "Conversion complete."
