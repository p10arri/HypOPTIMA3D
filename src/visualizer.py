
import os
import numpy as np
from typing import Optional
from pathlib import Path
from loguru import logger

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import seaborn as sns
import pandas as pd

import umap

from src.utils.hyperbolic_function import HyperbolicSpace
from src.utils.enums import Space, NineClassesLabel

class Visualizer:
    def __init__(self, space: Space, encoder_variant:Optional[str] = None, hyp_c: float = 1.0, simsiam=False, experiment:str = None):
        self.space = space
        self.encoder_variant = encoder_variant
        self.experiment = experiment if experiment else self._get_experiment_basename(simsiam)
        self.hyperbolic_space = HyperbolicSpace(c=hyp_c)

    def _get_experiment_basename(self, is_simsiam: bool) -> str:
        # Prefix for Training Mode
        prefix = "Simsiam" if is_simsiam else ""
        
        # Geometry Abbreviation
        geo_map = {
            Space.HYPERBOLIC: "Hyp",
            Space.SPHERICAL: "Sph",
            Space.EUCLIDEAN: "Euc"
        }
        geo = geo_map.get(self.space, "")
        return f"{prefix}{geo}{self.encoder_variant}"

    def plot_embeddings(self, embs: np.ndarray, labels: np.ndarray, path: str, title: str = "Embedding Projection"):
        """Main plotting entry point with support for Hyperbolic UMAP."""
        # Dimension reduction
        if embs.shape[1] > 2:
            if self.space == Space.HYPERBOLIC:
                reducer = umap.UMAP(
                    n_components=2, 
                    output_metric='hyperboloid',    # Output in Lorentz model
                    n_neighbors=50,                 # Broad view to see across the disk
                    min_dist=0.1, 
                    densmap=False,                  #  Non-Euclidean output metric not supported for densMAP
                    random_state=1337
                )
            elif self.space == Space.SPHERICAL:
                reducer = umap.UMAP(
                    n_components=2, 
                    metric='cosine',        # Critical for spherical input
                    n_neighbors=50,         # Broad view to see across the sphere
                    min_dist=0.3,           # Spread points out
                    densmap=True,           # Preserves local density better
                    random_state=1337
                )
            else:
                reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=1337)
            
            embs_2d = reducer.fit_transform(embs)
        else:
            embs_2d = embs

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Projections
        if self.space == Space.HYPERBOLIC:
            embs_2d = self.hyperbolic_space.project_hyperboloid_to_poincare(embs_2d)
            self._setup_circular_boundary(ax, is_poincare=True)
        
        if self.space == Space.SPHERICAL:
            # Get directions by unit-normalizing
            norms = np.linalg.norm(embs_2d, axis=1, keepdims=True)
            directions = embs_2d / (norms + 1e-8)
            # Scale radius by Min-Max normalization
            min_n, max_n = norms.min(), norms.max()
            scaled_radius = (norms - min_n) / (max_n - min_n + 1e-8)
            # reconstruct 2D coordinates
            embs_2d = directions * scaled_radius

            self._setup_circular_boundary(ax, is_poincare=False)

        # Convert labels to class names
        class_names_list = NineClassesLabel.to_name(labels)

        # Plot
        df_plot = pd.DataFrame({
            'x': embs_2d[:, 0],
            'y': embs_2d[:, 1],
            'Label': [name.upper() for name in class_names_list]
        })
        scatter_plot = sns.scatterplot(
            data=df_plot, x='x', y='y', hue='Label', 
            palette='tab10', s=20, alpha=0.8, edgecolor='none', ax=ax
        )
        
        subtitle = {
            Space.HYPERBOLIC: "Poincare Disk",
            Space.SPHERICAL: "Spherical Surface",
            Space.EUCLIDEAN: "Euclidean Plane"
        }.get(self.space, "Projection")

        ax.set_title(f"{title}\n for {self.experiment} onto {subtitle}", fontsize=14, fontweight='bold')
        
        if self.space == Space.EUCLIDEAN:
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.axis("off") 

        ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_embeddings_amd_grouped(self, embs: np.ndarray, labels: np.ndarray, path: str, title: str = "Embedding Projection (Grouped AMD vs. Other Pathologies)"):
        """Main plotting entry point with support for Hyperbolic UMAP. Age-Related Macroma Degeneracy are grouped"""

        # Dimension reduction
        if embs.shape[1] > 2:
            if self.space == Space.HYPERBOLIC:
                reducer = umap.UMAP(
                    n_components=2, 
                    output_metric='hyperboloid',    # Output in Lorentz model
                    n_neighbors=50,                 # Broad view to see across the disk
                    min_dist=0.1, 
                    densmap=False,                  #  Non-Euclidean output metric not supported for densMAP
                    random_state=1337
                )
            elif self.space == Space.SPHERICAL:
                reducer = umap.UMAP(
                    n_components=2, 
                    metric='cosine',        # Critical for spherical input
                    n_neighbors=50,         # Broad view to see across the sphere
                    min_dist=0.3,           # Spread points out
                    densmap=True,           # Preserves local density better
                    random_state=1337
                )
            else:
                reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=1337)
            
            embs_2d = reducer.fit_transform(embs)
        else:
            embs_2d = embs

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Projections
        if self.space == Space.HYPERBOLIC:
            embs_2d = self.hyperbolic_space.project_hyperboloid_to_poincare(embs_2d)
            self._setup_circular_boundary(ax, is_poincare=True)
        
        if self.space == Space.SPHERICAL:
            # Get directions by unit-normalizing
            norms = np.linalg.norm(embs_2d, axis=1, keepdims=True)
            directions = embs_2d / (norms + 1e-8)
            # Scale radius by Min-Max normalization
            min_n, max_n = norms.min(), norms.max()
            scaled_radius = (norms - min_n) / (max_n - min_n + 1e-8)
            # reconstruct 2D coordinates
            embs_2d = directions * scaled_radius

            self._setup_circular_boundary(ax, is_poincare=False)

        # Convert labels to class names
        class_names_list = NineClassesLabel.to_name(labels)

        # Define the new class
        to_merge = ["CNV1", "CNV2", "CNV3", "GA", "IAMD"]
        new_class_names_list = []
        for name in class_names_list:
            name_upper = name.upper()
            if name_upper in to_merge:
                new_class_names_list.append("AMD")
            else:
                new_class_names_list.append(name_upper)
                
        # Plot
        df_plot = pd.DataFrame({
            'x': embs_2d[:, 0],
            'y': embs_2d[:, 1],
            'Label': [name.upper() for name in new_class_names_list]
        })


        scatter_plot = sns.scatterplot(
            data=df_plot, x='x', y='y',
            palette='tab10', s=20, alpha=0.8, edgecolor='none', ax=ax,
            hue='Label', # Color based in new_class_names_list
            style=class_names_list # Different shapes for all classes included to_merge classes
        )
        
        subtitle = {
            Space.HYPERBOLIC: "Poincare Disk",
            Space.SPHERICAL: "Spherical Surface",
            Space.EUCLIDEAN: "Euclidean Plane"
        }.get(self.space, "Projection")

        ax.set_title(f"{title}\n for {self.experiment} onto {subtitle}", fontsize=14, fontweight='bold')
        
        if self.space == Space.EUCLIDEAN:
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.axis("off") 

        ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def _setup_circular_boundary(self, ax, is_poincare: bool = False):
        """
        Draws a reference circle for geometry context.
        Hyperbolic: The boundary is at 1/sqrt(c).
        Spherical/Euclidean: The boundary is at 1.0.
        """
        radius = 1.0 / np.sqrt(self.hyperbolic_space.c) if is_poincare else 1.0

        color = 'black' if is_poincare else 'gray'
        style = '-' if is_poincare else '--'
        alpha = 1.0 if is_poincare else 0.5

        boundary = plt.Circle((0, 0), radius, color=color, fill=False, 
                               linewidth=2, linestyle=style, alpha=alpha, zorder=0)
        ax.add_patch(boundary)
        
        limit = radius * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axis("off")


    def _setup_poincare_disk(self, ax):
        """Draws the boundary of the Poincaré disk (Unit Circle)."""
        # The boundary in the Poincare disk is always 1.0 in standard mapping
        boundary = plt.Circle((0, 0), 1.0, color='black', fill=False, 
                             linewidth=2, linestyle='-', zorder=10)
        ax.add_patch(boundary)
        
        limit = 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
    
    def plot_confusion_matrix(self, cm_norm: np.ndarray, class_names: list, path: str):
        """Plots a confusion matrix using pre-normalized data."""
        # Ensure it's a DataFrame for better labeling
        df_cm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title(f"Normalized Confusion Matrix\n {self.experiment}", fontsize=14, fontweight='bold')
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_interclass_distances(self, embs: np.ndarray, labels: np.ndarray, class_names: list, path: str):
        """
        Plots a heatmap of average distances between classes.
        Uses the distance metric appropriate for the geometry.
        """
        n_classes = len(class_names)
        dist_matrix = np.zeros((n_classes, n_classes))
        
        # Pre-calculate centroids or handle distance-to-all for classes
        for i in range(n_classes):
            for j in range(n_classes):
                mask_i = (labels == i)
                mask_j = (labels == j)
                
                if not any(mask_i) or not any(mask_j):
                    continue
                
                # Representative distance: mean distance between all pairs or centroids
                # For efficiency in high-D, we use the distance between class centroids
                c_i = embs[mask_i].mean(axis=0)
                c_j = embs[mask_j].mean(axis=0)
                
                if self.space == Space.HYPERBOLIC:
                    dist_matrix[i, j] = self._poincare_distance(c_i, c_j)
                else:
                    dist_matrix[i, j] = np.linalg.norm(c_i - c_j)

        df_dist = pd.DataFrame(dist_matrix, index=class_names, columns=class_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_dist, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True)
        plt.title(f"Average Inter-Class Distances\n({self.space.name})", fontsize=14, fontweight='bold')
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    def _get_centroid(self, embs: np.ndarray) -> np.ndarray:
        """Calculate the geometric center of a point cloud"""
        if self.space == Space.EUCLIDEAN:
            return embs.mean(axis=0)

        if self.space == Space.SPHERICAL:
            # The centroid is the normalized direction of the mean
            mean_vec = embs.mean(axis=0)
            return mean_vec / (np.linalg.norm(mean_vec) + 1e-8)

        if self.space == Space.HYPERBOLIC:
            # Calculating the exact Fréchet mean is iterative, but the 
            # Einstein Midpoint is a very high-quality approximation for Poincare
            # It uses weights based on the Lorentz factor (gamma)
            sq_norms = np.sum(embs**2, axis=1)
            # Gamma factor: 1 / (1 - ||x||^2)
            weights = 1 / (1 - sq_norms + 1e-7)
            return np.average(embs, axis=0, weights=weights)
    
        return embs.mean(axis=0)

    def _poincare_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Helper to calculate hyperbolic distance between two points."""
        sq_dist = np.sum((u - v)**2)
        u_norm = np.sum(u**2)
        v_norm = np.sum(v**2)
        
        # Hyperbolic distance formula for Poincare Disk (c=1.0)
        arg = 1 + 2 * sq_dist / ((1 - u_norm) * (1 - v_norm) + 1e-7)
        return np.arccosh(np.clip(arg, 1.0, 1e7))

    def _load_baseline_cm(self, path: Path) -> np.ndarray:
        """
        Loads the Euclidean baseline confusion matrix of the specific encoder version.
        """
        
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found at {path}. "
                                    f"Ensure you ran the Euclidean experiment first.")
        
        logger.info(f"Loading baseline confusion matrix from {path}")
        return np.load(path)

    def plot_delta_confusion_matrix(
        self, 
        cm_baseline_path: Path, 
        cm_experimental: np.ndarray, 
        class_names: list, 
        path: str,
        title: Optional[str] = None,
    ):
        """
        Plots the difference between the baseline (Euclidean) and experimental Confusion Matrix (CM).
        
        Logic: 
        - Diagonal: Blue means the experimental model is MORE accurate.
        - Off-Diagonal: Red means the experimental model is LESS confused.
        """
        cm_baseline = self._load_baseline_cm(cm_baseline_path)
        
        # Space check
        if self.space == Space.EUCLIDEAN:
            logger.warning("Generating a Delta CM for Euclidean vs Euclidean will result in all zeros.")

        # Calculate delta
        if not np.allclose(cm_experimental.sum(axis=1), 1.0, atol=1e-2):
            print("not np.allclose(cm_experimental.sum(axis=1), 1.0, atol=1e-2)")
            cm_experimental = cm_experimental.astype('float') / (cm_experimental.sum(axis=1)[:, np.newaxis] + 1e-8)

        delta = cm_experimental - cm_baseline
        df_delta = pd.DataFrame(delta, index=class_names, columns=class_names)
        
        plt.figure(figsize=(12, 10))
        
        # Colormap: RdBu (Red-White-Blue)
        # Positive (Blue) = Experimental is higher (Good on diagonal)
        # Negative (Red) = Experimental is lower (Good on off-diagonal)
        sns.heatmap(
            df_delta, 
            annot=True, 
            fmt=".2f", 
            cmap="RdBu", 
            center=0,
            cbar_kws={'label': 'Difference (Experimental - Baseline)'}
        )
        
        default_title = f"Geometric Gain for {self.experiment}: {self.space.name} vs EUCLIDEAN"
        plt.title(title or default_title, fontsize=14, fontweight='bold')
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                    "Interpretation: Blue on diagonal = Improvement | Red off-diagonal = Reduced Confusion", 
                    ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    
    def plot_class_accuracies(self, csv_path: str, output_path: str, title: str = "Class Accuracy Comparison"):
        """
        Parses the evaluation_summary.csv and plots a grouped bar chart 
        using NineClassesLabel enum for professional naming.
        """
        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found at {csv_path}. Skipping plot.")
            return

        # Load the data
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning("Summary CSV is empty!")
            return

        
        # Identify columns that contain class accuracies
        class_acc_cols = [c for c in df.columns if "accuracy" in c and "class_" in c]
        
        plot_df = df.melt(
            id_vars=['experiment_name'], 
            value_vars=class_acc_cols,
            var_name='ClassID', 
            value_name='Accuracy'
        )
        
        # Extract the integer ID from "class_0_accuracy" -> 0
        plot_df['id'] = plot_df['ClassID'].str.extract('(\d+)').astype(int)
        
        # Map IDs to Names
        plot_df['Class Name'] = plot_df['id'].apply(lambda x: NineClassesLabel.to_name(x)[0])

        # Initialize the plot
        plt.figure(figsize=(20, 8))
        sns.set_style("whitegrid")
        
        # Create grouped bar chart
        ax = sns.barplot(
            data=plot_df, 
            x='Class Name', 
            y='Accuracy', 
            hue='experiment_name',
            palette='viridis'
        )

        # Formatting
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Accuracy Score", fontsize=12)
        plt.xlabel("Class", fontsize=12)
        plt.ylim(0, 1.15) 
        plt.xticks(rotation=15) # Slight rotation for better readability of labels
        
        # Legend outside the plot
        plt.legend(title='Experiment', bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        
        # Add value annotations on top of bars
        for p in ax.patches:
            height = p.get_height()
            
            if height > 0:
                ax.annotate(f'{height:.2f}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', 
                            va='bottom', 
                            xytext=(0, 3), # vertical offset
                            textcoords='offset points',
                            fontsize=8, 
                            fontweight='bold',
                            rotation=90,
                            clip_on=True,
                            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Class accuracy comparison saved to {output_path}")

    def plot_roc_auc_radar(self, csv_path: str, output_path: str, title: str = "Pathologic ROC Curve"):
        """
        Generates a radar chart comparing ROC AUC across all classes for each experiment.
        """
        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found at {csv_path}. Skipping radar plot.")
            return

        df = pd.read_csv(csv_path)
        if df.empty:
            return

        # Map ROC AUC columns
        roc_cols = sorted([c for c in df.columns if "roc_auc_class_" in c])
        labels = [NineClassesLabel.to_name(int(c.split('_')[-1]))[0].upper() for c in roc_cols]
        num_vars = len(labels)

        # Compute angles for the radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each experiment
        colors = plt.cm.get_cmap("Set1", len(df))

        for i, (idx, row) in enumerate(df.iterrows()):
            values = row[roc_cols].values.flatten().tolist()
            values += values[:1] # Close the loop
            
            exp_name = row['experiment_name']
            ax.plot(angles, values, color=colors(i), linewidth=2, label=exp_name)
            ax.fill(angles, values, color=colors(i), alpha=0.15)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')

        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], color="grey", size=8)
        
        plt.title(title, size=16, fontweight='bold', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Radar chart saved to {output_path}")