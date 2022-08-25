#!/bin/bash

echo -e "$$\t${BASH_SOURCE[0]}" >> pids.txt
python train_agent.py maxcut cycle