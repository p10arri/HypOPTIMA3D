import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from torch.utils.data import DataLoader
from scipy import stats

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, roc_auc_score, confusion_matrix
from src.utils.enums import Space, TrainingMode, NineClassesLabel

from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.sphere import Sphere

class Evaluator:
    def __init__(
        self, 
        device: torch.device, 
        space: Space,
        training_mode: TrainingMode,
        hyp_c: float,
        k_list: List[int] = [1, 2, 4, 8, 16, 32]
    ):
        self.device = device
        self.space = space
        self.training_mode = training_mode
        self.hyp_c = hyp_c
        self.k_list = k_list

    @torch.no_grad()
    def run(self, model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> Dict[str, Any]:
        model.eval()

        all_outputs = []
        all_targets = []
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                x, targets = batch[0], batch[1]
            else:
                x = batch["img"]
                targets = batch["label"]

            x = x.to(self.device)
            targets = targets.to(self.device)

            outputs = model(x)

            if self.training_mode == TrainingMode.SUPERVISED:
                out = outputs['logits']
                loss,_ = loss_fn(out, targets)

            elif self.training_mode == TrainingMode.CONTRASTIVE:
                out = outputs['embeddings']
                # Recall@1 is the target in validation stage
                loss = torch.tensor(0.0).to(self.device) # Placeholder

            else:
                raise NotImplementedError(f"Evaluation run not implemented for TrainingMode:{self.training_mode}")
            
            
            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            all_outputs.append(out)
            all_targets.append(targets)

        avg_loss = total_loss / n_samples
        eval_metrics = {"eval/loss": avg_loss}
        # concatenate
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        

        if self.training_mode == TrainingMode.SUPERVISED:
            sup_metrics = self._evaluate_supervised_batch(all_outputs, all_targets)
            eval_metrics.update(sup_metrics)
        else:
            cont_metrics = self._evaluate_contrastive_batch(all_outputs, all_targets)
            eval_metrics.update(cont_metrics)

        return eval_metrics

    def _evaluate_supervised_batch(self, logits, target)-> Dict[str, Any]:

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        y_true = target.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs.cpu().numpy()

        metrics = self._compute_per_class_metrics(y_true, y_pred)
        metrics.update(self._compute_global_metrics(y_true, y_prob))

        # Add raw data for WandB plotting
        metrics["raw_y_true"] = y_true
        metrics["raw_y_pred"] = y_pred
        
        return metrics

    def _evaluate_contrastive_batch(self, embeddings, targets) -> Dict[str, Any]:
        # num_classes = len(NineClassesLabel.class_ids()) # Not compatible with OCTMNIST 
        unique_labels = torch.unique(targets)
        num_classes = int(unique_labels.max().item() + 1)
        
        print(f" [DEBUG]: Number of classes found in the evaluation stage: {num_classes}")

        y_true = targets.cpu().numpy()

        metrics = {}
        
        # Calculate accuracy for different K values
        for k in self.k_list:
            current_k = min(k,len(embeddings)-1)

            if current_k < 1:
                continue

            # compare embeddigns against themselves
            pred_indices, _ = self.knn_predict(
                queries=embeddings.to(self.device),
                bank=embeddings.to(self.device),
                bank_labels=targets.to(self.device),
                classes=num_classes,
                knn_k=current_k,
                knn_t=0.1,
                space=self.space,
                hyp_c=self.hyp_c,
                is_self_comparison=True
            )

            # Top1 class prediction
            top1_pred = pred_indices[:, 0].cpu().numpy()
            metrics[f"eval/knn_acc@{k}"] = accuracy_score(y_true, top1_pred)

        # Include raw recall
        recall_results = self._recall_dict(embeddings, targets, return_dict=True)
        metrics.update({f"eval/recall@{k}": v for k, v in recall_results.items()})

        return metrics

    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        per_class_recall = recall_score(y_true, y_pred, average=None)

        return {
            "eval/accuracy": accuracy_score(y_true, y_pred),
            "eval/f1_macro": f1_score(y_true, y_pred, average="macro"),
            "eval/f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "eval/recall_per_class": per_class_recall,
            "eval/confusion_matrix": cm
        }

    def _compute_global_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        try:
            present_classes = np.unique(y_true)
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=present_classes)
        except Exception:
            auc = 0.0
        return {"eval/auc": auc}
    
    def _recall_dict(self, x_emb: torch.Tensor, y_labels: torch.Tensor, return_dict: bool = False) -> Union[float, Dict[str, float]]:
        x_emb = x_emb.to(self.device)
        
        # Compute Distance Matrix
        if self.space == Space.HYPERBOLIC:
            manifold = Lorentz(k=self.hyp_c)
            # x_emb is [N, D+1]. dist requires [N, 1, D+1] vs [1, N, D+1] for pairwise
            # Result is [N, N]. Negative because .topk(largest=True) expects similarity.
            dist_m = -manifold.dist(x_emb.unsqueeze(1), x_emb.unsqueeze(0))

        elif self.space == Space.SPHERICAL:
            manifold = Sphere()
            dist_m = -manifold.dist(x_emb.unsqueeze(1), x_emb.unsqueeze(0))

        else: # Space.EUCLIDEAN
            x_norm = nn.functional.normalize(x_emb, p=2, dim=1)
            dist_m = torch.mm(x_norm, x_norm.t())

        # Prevent self-matching (diagonal)
        dist_m.fill_diagonal_(-float('inf'))
        
        # Retrieve Top-K indices
        max_k = min(len(x_emb) - 1, max(self.k_list))
        _, topk_indices = dist_m.topk(max_k, dim=1, largest=True)
        
        # Compute Recall@K
        y_labels = y_labels.view(-1).cpu() 
        topk_labels = y_labels[topk_indices.cpu()] # [N, max_k]
        
        recall_results = {}
        for k in self.k_list:
            actual_k = min(k, topk_labels.size(1))
            if actual_k <= max_k:
                # Sliced neighbors: [N, actual_k]
                current_topk = topk_labels[:, :actual_k]
                
                # Ground truth labels expanded to match neighbors: [N, actual_k]
                gt_expanded = y_labels.view(-1, 1).expand(-1, actual_k)
                
                # Check for matches along the neighbor dimension
                correct = (gt_expanded == current_topk).any(dim=1)
                recall_results[f"recall@{k}"] = correct.float().mean().item()

        return recall_results if return_dict else recall_results.get("recall@1", 0.0)
    
    @staticmethod
    def knn_predict(
        queries: torch.Tensor, 
        bank: torch.Tensor, 
        bank_labels: torch.Tensor, 
        classes: int, 
        knn_k: int, 
        knn_t: float, 
        space: Space, 
        hyp_c: float = 1.0, 
        is_self_comparison: bool = True
    ):
        """
        Predicts labels using kNN with manifold-aware distance metrics.
        """
        max_label = bank_labels.max().item()
        actual_num_classes = max(classes, int(max_label + 1))
        print(f"DEBUG: Matrix width: {actual_num_classes}, Max label in bank: {max_label}")

        if space == Space.HYPERBOLIC:
            manifold = Lorentz(k=hyp_c)
            # Lorentz.dist handles the Minkowski inner product logic internally
            # Negative distance because we want smallest distance = highest similarity
            dist = manifold.dist(queries.unsqueeze(1), bank.unsqueeze(0))
            sim_matrix = -dist 

        elif space == Space.SPHERICAL:
            manifold = Sphere()
            # Geodesic distance on the sphere
            dist = manifold.dist(queries.unsqueeze(1), bank.unsqueeze(0))
            sim_matrix = -dist

        else:  # Space.EUCLIDEAN
            # Standard Cosine Similarity for Euclidean space
            q_norm = nn.functional.normalize(queries, p=2, dim=1)
            b_norm = nn.functional.normalize(bank, p=2, dim=1)
            sim_matrix = torch.mm(q_norm, b_norm.t())

        # Mask Diagonal (avoid predicting self)
        if is_self_comparison:
            sim_matrix.fill_diagonal_(-float('inf'))

        # Get Top-K neighbors
        # Since negative distances, 'largest=True' gives the closest points
        sim_weight, sim_indices = sim_matrix.topk(k=min(knn_k, sim_matrix.size(1)), dim=-1, largest=True)
        
        # ensure bank_labels are on same device and expanded
        bank_labels_expanded = bank_labels.view(1, -1).expand(queries.size(0), -1)

        # Get Labels of neighbors
        # bank_labels: [N], sim_indices: [B, K] -> sim_labels: [B, K]
        max_idx = bank_labels.size(0) - 1
        if sim_indices.max() > max_idx:
            sim_indices = torch.clamp(sim_indices, 0, max_idx)
            
        sim_labels = torch.gather(bank_labels_expanded, dim=-1, index=sim_indices)

        # sim_labels = torch.gather(bank_labels.expand(queries.size(0), -1), dim=-1, index=sim_indices)
        
    
        # Convert distances/similarities into weights
        sim_weight = (sim_weight / knn_t).exp()
        

        one_hot_label = torch.zeros(queries.size(0) * sim_indices.size(1), actual_num_classes, device=queries.device)
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        
        weighted_votes = one_hot_label.view(queries.size(0), -1, actual_num_classes) * sim_weight.unsqueeze(dim=-1)
        pred_scores = torch.sum(weighted_votes, dim=1)
        
        return pred_scores.argsort(dim=-1, descending=True), pred_scores

    @staticmethod
    def _perform_mann_whitney_tests(embs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        unique_classes = np.unique(labels)
        norms = np.linalg.norm(embs, axis=1)
        results = {}
        
        for c in unique_classes:
            pos_norms = norms[labels == c]
            neg_norms = norms[labels != c]
            if len(pos_norms) > 0 and len(neg_norms) > 0:
                _, p_val = stats.mannwhitneyu(pos_norms, neg_norms, alternative='two-sided')
                results[f"eval/stat_p_class_{c}"] = float(p_val)
        return results