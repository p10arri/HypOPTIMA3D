#!/bin/bash

# Exit immediately if a command fails
set -e

echo "🚀 Starting TEST Experiment Suite..."
echo "📂 All dev logs will be mapped to: results/dev_trash"

# Configurations
SPACES=("euclidean" "spherical" "hyperbolic")
AUGMENTATIONS=("normal" "heavy" "oct_classifier")
MODES=("supervised" "contrastive")
SKIP_HEADS=("False" "True")

# Loop through all combinations
for SPACE in "${SPACES[@]}"; do
    for AUG in "${AUGMENTATIONS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SKIP in "${SKIP_HEADS[@]}"; do

                # Determine Namwe
                if [ "$SKIP" == "True" ]; then TOKEN="EmbeddingToken"; else TOKEN="ClassToken"; fi
                
                EXP_NAME="TEST_${SPACE}_${AUG}_${MODE}_${TOKEN}"
                
                echo "------------------------------------------------"
                echo "⚡ Running: $EXP_NAME"
                echo "------------------------------------------------"

                # Parameters
                CMD="python main.py \
                    experiment=$EXP_NAME \
                    ++fast_dev_run=True \
                    ++wandb.mode=disabled \
                    space=$SPACE \
                    augmentations=$AUG \
                    training_mode=$MODE \
                    model.skip_class_head=$SKIP\
                    hydra.run.dir=results/dev_trash/$EXP_NAME"

                # Add specific sampling for Contrastive mode to prevent batch errors
                if [ "$MODE" == "contrastive" ]; then
                    CMD="$CMD ++data.sampling.m_per_class=1"
                fi

                eval $CMD

                echo -e "✅ Finished: $EXP_NAME\n"

            done
        done
    done
done

echo "🎉 All experiments completed successfully!"