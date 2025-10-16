#!/bin/bash

# Create directory structure for Argoverse 2
mkdir -p dataset/argoverse2
cd dataset/argoverse2

echo "======= Downloading Argoverse 2 Motion Forecasting Dataset ======="

# Argoverse 2 Download Links (Motion Forecasting)
wget https://argoverse-hd.s3.us-east-2.amazonaws.com/av2_mf_train.tar
wget https://argoverse-hd.s3.us-east-2.amazonaws.com/av2_mf_val.tar
# Test split is private for leaderboard submission, only available via official site

echo "======= Extracting Files ======="
tar -xf av2_mf_train.tar
tar -xf av2_mf_val.tar

echo "======= Dataset Ready at dataset/argoverse2 ======="