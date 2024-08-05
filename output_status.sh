#!/bin/bash

while true; do
    clear
    echo "$(date +'%T')"

    # Get the list of processes that match 'counter'
    process_list=$(ps -ef | grep counter)

    # Filter the list for processes that have 'python -m' in them
    filtered_list=$(echo "$process_list" | grep 'python -m')

    # Count the number of matching processes
    count=$(echo "$filtered_list" | wc -l)

    echo "Number of 'python -m' processes: $count"

    # Loop through each folder in the current directory that starts with "output_"
    for dir in output_*/; do
        # Check if the item is a directory
        if [ -d "$dir" ]; then
            # Count the number of files in the directory
            file_count=$(ls -1q "$dir" | wc -l)
            # Print the number of files in the directory
            echo -e "$file_count\tfiles in ${dir%/}"
        fi
    done

    # Wait for 10 seconds before updating again
    sleep 10
done