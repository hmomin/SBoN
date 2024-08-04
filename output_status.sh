#!/bin/bash

while true; do
    clear
    echo "$(date)"

    # Loop through each folder in the current directory that starts with "output_"
    for dir in output_*/; do
        # Check if the item is a directory
        if [ -d "$dir" ]; then
            # Count the number of files in the directory
            file_count=$(ls -1q "$dir" | wc -l)
            # Print the number of files in the directory
            echo "$file_count files in ${dir%/}..."
        fi
    done

    # Wait for 10 seconds before updating again
    sleep 10
done