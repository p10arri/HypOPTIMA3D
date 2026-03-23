#!/bin/bash

# Exit immediately if a command fails
set -e

### ----------------- RUN MULTIPLE EXPERIMENT -----------------

echo "🚀 Starting Experiment Suite..."

# Configurations
SPACES=("spherical" "hyperbolic")
AUGMENTATIONS=("heavy")
MODES=("contrastive")
SKIP_HEADS=("False")

# ---- All combinations ----
# SPACES=("euclidean" "spherical" "hyperbolic")
# AUGMENTATIONS=("normal" "heavy" "oct_classifier")
# MODES=("supervised" "contrastive")
# SKIP_HEADS=("False" "True")




# Loop through all combinations
for SPACE in "${SPACES[@]}"; do
    for AUG in "${AUGMENTATIONS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SKIP in "${SKIP_HEADS[@]}"; do

                # Determine Namwe
                if [ "$SKIP" == "True" ]; then TOKEN="EmbeddingToken"; else TOKEN="ClassToken"; fi
                
                EXP_NAME="${SPACE}_${AUG}_${MODE}_${TOKEN}"
                
                echo "------------------------------------------------"
                echo "⚡ Running: $EXP_NAME"
                echo "------------------------------------------------"

                # Parameters
                CMD="python main.py \
                    experiment=$EXP_NAME \
                    ++fast_dev_run=False \
                    ++wandb.mode=online \
                    space=$SPACE \
                    augmentations=$AUG \
                    training_mode=$MODE \
                    model.skip_class_head=$SKIP \
                    trainer.epochs=100 \
                    trainer.eval_frequency=5 \
                    data.batch_size=32"

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