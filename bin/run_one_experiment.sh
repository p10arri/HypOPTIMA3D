#!/bin/bash

# Exit immediately if a command fails
set -e

### ----------------- RUN ONE EXPERIMENT -----------------
echo "🚀 Starting One Experiment Suite..."

python main.py experiment=EucViT_Pretrained_Normal_Contrastive_ClassToken \
    ++fast_dev_run=False \
    ++trainer.resume_from=False\
    space=euclidean \
    augmentations=heavy \
    model=vit3d \
    model.skip_class_head=False \
    model.pretrained=True \
    ++training_mode=supervised \
    ++optimizer.lr=0.0001
echo "🎉 Experiment completed successfully!"