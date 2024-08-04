#!/bin/bash

# kills all job hoggers
pkill -f "^python.*job_hogger.*"
pkill -f "^python.*counterfactual.*"