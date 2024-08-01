#!/bin/bash

# ensure a commit message is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <commit_message>"
    exit 1
fi
commit_message="$1"

black .
git add .
git commit -m "$commit_message"
git push
