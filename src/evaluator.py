from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Union, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from src.utils.hyperbolic_function import HyperbolicSpace
from src.utils.enums import Space


class Evaluator:
    def __init__(
        self, 
        device: torch.device, 
        space: Space = Space.EUCLIDEAN,
        hyp_c: float = 0.0, 
        k_list: List[int] = [1, 2, 4, 8, 16, 32]
    ):
        self.device = device
        self.space = space
        self.hyp_c = hyp_c
        self.k_list = k_list

    @torch.no_grad()
    def get_embeddings(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        skip_head: bool = False,
        show_progress: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        logger.debug(f"Extracting embeddings | skip_head={skip_head}")
        model.eval()

        all_embs = []
        all_labels = []

        loader = tqdm(dataloader) if show_progress else dataloader

        for batch in loader:
            images = batch["img"]
            labels = batch["label"]

            if isinstance(images, list):
                images = images[0] # only use the first view

            images = images.to(self.device)
            emb = model(images, skip_head=skip_head)

            all_embs.append(emb.cpu())
            all_labels.append(labels.cpu())

        embeddings = torch.cat(all_embs, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # DEBUG: Ensure we got the whole dataset
        logger.debug(f"Collected {embeddings.shape[0]} embeddings for evaluation")

        return embeddings, labels

    @staticmethod
    def embedding_basic_stats(emb: torch.Tensor) -> Dict[str, float]:
        """Computes basic statistics of embedding norms."""
        emb_norm = torch.norm(emb, dim=-1, keepdim=True)
        return {
            "N": emb.shape[0],
            "D": emb.shape[1],
            "norm_mean": emb_norm.mean().item(),
            "norm_std": emb_norm.std().item(),
            "norm_min": emb_norm.min().item(),
            "norm_max": emb_norm.max().item(),
        }

    @staticmethod
    def _recall_at_k(T: torch.Tensor, Y: torch.Tensor, k: int) -> float:
        """Vectorized computation of recall@k."""
        T_expanded = T.view(-1, 1).expand(-1, k)        # [num_samples, k]
        Y_topk = Y[:, :k]                               # [num_samples, k]
        correct = (T_expanded == Y_topk).any(dim=1).float()  # [num_samples]
        return correct.mean().item()

    def _recall_dict(
        self, 
        x_emb: torch.Tensor, 
        y_emb: torch.Tensor, 
        return_dict: bool = False
    ) -> Union[float, List[float]]:
        """Calculates recall@k for the k_list using space-specific distance metrics."""
        
        x_emb = x_emb.to(self.device)
        if self.space == Space.HYPERBOLIC:
            hyp_space = HyperbolicSpace(c=self.hyp_c)
            # dist_m = torch.stack([-hyp_space.distance(x_emb[i : i + 1], x_emb) for i in range(len(x_emb))])
            dist_m = -hyp_space.distance(x_emb, x_emb)
        
        else:
            # cosine similarity (dot product of normalized embs)
            x_norm = nn.functional.normalize(x_emb, p=2, dim=1)
            dist_m = x_norm @ x_norm.t()
            

        
        dist_m = dist_m.cpu()
        y_emb = y_emb.cpu() 
        
        # Ensure no more neighbours than available are requested
        max_k = min(len(x_emb)-1, max(self.k_list))
        _, topk_indices = dist_m.topk(1 + max_k, largest=True)
        y_cur = y_emb[topk_indices[:, 1:]]

        # recall_list = [self._recall_at_k(y_emb, y_cur, k) for k in self.k_list]
        recall_dict = {f"recall@{k}": self._recall_at_k(y_emb, y_cur, k) for k in self.k_list}
        return recall_dict if return_dict else recall_dict["recall@1"]

    @torch.no_grad()
    def evaluate_recall(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Standard Recall@K evaluation."""
        logger.debug("Running Recall@K evaluation")
        emb, labels = self.get_embeddings(model, dataloader)
        recalls = self._recall_dict(emb, labels, return_dict=True)

        return {f"recall@{k}": r for k, r in zip(self.k_list, recalls)}
    
    @torch.no_grad()
    def evaluate_head_body(self, model: nn.Module, dataloader: DataLoader):
        """Compares recall between the penultimate layer and the final output layer."""
        logger.debug("Running recall head-body evaluation")

        emb_head, labels = self.get_embeddings(model, dataloader, skip_head=False)
        emb_body, _ = self.get_embeddings(model, dataloader, skip_head=True)

        recall_head = self._recall_dict(emb_head, labels, return_dict=True)
        recall_body = self._recall_dict(emb_body, labels, return_dict=True)

        return recall_head, recall_body, emb_head, emb_body

    @staticmethod
    def _topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[torch.Tensor]:
        """Computes top-k accuracy for categorical classification."""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    @torch.no_grad()
    def evaluate_topk(self, model: nn.Module, dataloader: DataLoader, topk=(1, 5)) -> Dict[str, float]:
        """Standard classification Top-K accuracy."""
        logger.debug("Running topk evaluation")
        model.eval()

        correct = {k: 0.0 for k in topk}
        total = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            logits = model(x)
            accs = self._topk_accuracy(logits, y, topk=topk)

            batch_size = x.size(0)
            for k, acc in zip(topk, accs):
                correct[k] += acc.item() * batch_size / 100.0

            total += batch_size
        return {f"top{k}": 100.0 * correct[k] / total for k in topk}

    @staticmethod
    @torch.no_grad()
    def knn_predict(
        queries: torch.Tensor,
        bank: torch.Tensor,
        bank_labels: torch.Tensor,
        classes: int,
        knn_k: int,
        knn_t: float,
        space: Space = Space.EUCLIDEAN,
        hyp_c: float = 0.0,
        is_self_comparison: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts labels using kNN with geometry-aware distance metrics.
        """
        # Compute Similarity Matrix based on Geometry
        if space == Space.HYPERBOLIC:
            hyp = HyperbolicSpace(c=hyp_c)
            # Distance -> Similarity: exp(-dist)
            dist_m = hyp.distance(queries, bank)
            sim_matrix = torch.exp(-dist_m)
        elif space == Space.SPHERICAL:
            # Cosine similarity
            q_norm = nn.functional.normalize(queries, p=2, dim=1)
            b_norm = nn.functional.normalize(bank, p=2, dim=1)
            sim_matrix = torch.mm(q_norm, b_norm.t())
        else:
            # Euclidean dot product (on normalized embeddings)
            q_norm = nn.functional.normalize(queries, p=2, dim=1)
            b_norm = nn.functional.normalize(bank, p=2, dim=1)
            sim_matrix = torch.mm(q_norm, b_norm.t())

        # Mask Diagonal to prevent Data Leakage
        if is_self_comparison:
            eye_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device) * -1e9
            sim_matrix = sim_matrix + eye_mask

        # Weighted kNN
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        sim_labels = torch.gather(bank_labels.expand(queries.size(0), -1), dim=-1, index=sim_indices)
        
        # Temperature scaling
        sim_weight = (sim_weight / knn_t).exp()
        
        # Aggregate votes
        one_hot_label = torch.zeros(queries.size(0) * knn_k, classes, device=queries.device)
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        
        pred_scores = torch.sum(
            one_hot_label.view(queries.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )
        
        return pred_scores.argsort(dim=-1, descending=True), pred_scores

    @torch.no_grad()
    def evaluate_knn(
        self,
        model: nn.Module,
        memory_loader: DataLoader,
        test_loader: DataLoader,
        classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
    ) -> Dict[str, float]:
        """kNN Monitor evaluation for self-supervised learning."""
        logger.debug("Running kNN evaluation")
        model.eval()

        # Build Feature Bank
        feature_bank = []
        feature_labels = []
        for x, y in memory_loader:
            if isinstance(x, list): x = torch.stack(x)
            x = x.to(self.device)
            feature = model(x)
            # feature = nn.functional.normalize(feature, dim=1) #TODO: test for HypVitNormal
            feature_bank.append(feature)
            feature_labels.append(y)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous() 
        feature_labels = torch.cat(feature_labels, dim=0).to(self.device)

        # Evaluate
        total, top1, top5 = 0.0, 0.0, 0.0
        for x, y in test_loader:
            if isinstance(x, list): x = torch.stack(x)
            x, y = x.to(self.device), y.to(self.device)
            
            feature = model(x)
            feature = nn.functional.normalize(feature, dim=1)

            pred_labels,_ = self.knn_predict(
                feature, feature_bank, feature_labels, classes, knn_k, knn_t
            )

            total += y.size(0)
            top1 += (pred_labels[:, 0] == y).float().sum().item()
            top5 += (pred_labels[:, :5] == y.unsqueeze(dim=-1)).any(dim=-1).float().sum().item()
        
        return {"knn-top1": 100 * top1 / total, "knn-top5": 100 * top5 / total}
    
    @torch.no_grad()
    def evaluate_full_diagnostics(self, model: nn.Module, dataloader: DataLoader, is_baseline: Bool = False) -> Dict[str, Any]:
        logger.info(f"Executing full diagnostic for {self.space.name}...")

        
        logits, embs, labels = self._collect_results(model, dataloader)
        num_classes = len(torch.unique(labels))
        
        if is_baseline:

            # Logits to probabilities
            probs = torch.softmax(logits, dim=1) # logits: [Batch,Classes]
            # Logits to hard predictions
            preds = torch.argmax(logits, dim=1)

            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()          
            
            # Compute Recall
            # Backbones embeddings are used for recall calculation
            res_body = self._recall_dict(embs, labels, return_dict=True)
            recall_body = {f"body_{k}": v for k, v in res_body.items()} 
            recall_head = {f"head_{k}": 0.0 for k, v in res_body.items()} # Placeholder for csv file

            # Compute Mann-Whitney Test
            stat_embs = embs.cpu().numpy()             
            statistical_tests = self._perform_mann_whitney_tests(stat_embs, labels_np)


        else:
            # Prepare for kNN (Move to device)
            embs_dev = embs.to(self.device)
            labels_dev = labels.to(self.device)

            # Generate predictions via SpacekNN
            pred_indices, knn_scores = self.knn_predict(
                queries=embs_dev,
                bank=embs_dev,
                bank_labels=labels_dev,
                classes=num_classes,
                knn_k=20,
                knn_t=0.1,
                space=self.space,
                hyp_c=self.hyp_c,
                is_self_comparison=True # is_self_comparison=True vital for using the same set for bank/query
            )

            # Normalize scores to get valid probabilities for ROC AUC
            knn_probs = knn_scores / (knn_scores.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Prepare numpy arrays for sklearn metrics
            probs_np = knn_probs.cpu().numpy()
            preds_np = pred_indices[:, 0].cpu().numpy()
            labels_np = labels.cpu().numpy()
                  
            # Compute Recall
            res_head = self._recall_dict(logits, labels, return_dict=True)
            recall_head = {f"head_{k}": v for k, v in res_head.items()}

            res_body = self._recall_dict(embs, labels, return_dict=True)
            recall_body = {f"body_{k}": v for k, v in res_body.items()}
        
            # Compute Mann-Whitney Test
            stat_embs = logits.cpu().numpy()             
            statistical_tests = self._perform_mann_whitney_tests(stat_embs, labels_np)

        # Compute Metrics
        class_stats = self._compute_per_class_metrics(labels_np, preds_np)
        global_stats = self._compute_global_metrics(labels_np, probs_np)
        per_class_roc = self._compute_per_class_roc_auc(labels_np, probs_np) 
        
        return {
            **recall_head, 
            **recall_body,
            **per_class_roc,
            **class_stats,
            **global_stats,
            **statistical_tests,
            "raw": {"embs": embs, "labels": labels, "logits": logits}
        }

    def _collect_results(self, model: nn.Module, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect logits, embeddings, and labels."""
        model.eval()
        all_logits, all_embs, all_labels = [], [], []

        for batch in dataloader:
            x = batch["img"].to(self.device)
            y = batch["label"]
            
            logits = model(x, skip_head=False)
            embs = model(x, skip_head=True)

            all_logits.append(logits.cpu())
            all_embs.append(embs.cpu())
            all_labels.append(y.cpu())

        return torch.cat(all_logits), torch.cat(all_embs), torch.cat(all_labels)

    @staticmethod
    def _compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculates granular metrics for each class."""
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        
        # Avoid division by zero for empty classes
        per_class_acc = np.divide(cm.diagonal(), cm.sum(axis=1), 
                                  out=np.zeros_like(cm.diagonal(), dtype=float), 
                                  where=cm.sum(axis=1) != 0)
        
        mean_acc = (y_true == y_pred).mean()
        per_class_recall = recall_score(y_true, y_pred, average=None)
        
        return {
            "accuracy_per_class": per_class_acc,
            "mean_accuracy": mean_acc,
            "recall_per_class": per_class_recall,
            "confusion_matrix": cm
        }

    @staticmethod
    def _compute_global_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Calculates overall model performance metrics with label alignment."""
        try:
            # Filter classes to avoid the column mismatch
            present_classes = np.unique(y_true)
            if len(present_classes) < 2:
                return {"mean_roc_auc": 0.0}
            
            m_roc = roc_auc_score(
                y_true, 
                y_probs, 
                multi_class='ovr', 
                average='macro',
                labels=present_classes 
            )
        except Exception as e:
            logger.warning(f"ROC calculation skipped: {e}")
            m_roc = 0.0
            
        return {"mean_roc_auc": m_roc}

    @staticmethod
    def _perform_mann_whitney_tests(embs: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Performs Mann-Whitney U tests to determine if intra-class embedding 
        magnitudes differ significantly from inter-class magnitudes.
        """
        unique_classes = np.unique(labels)
        results = {}
        
        # Pre-calculate norms for efficiency
        norms = np.linalg.norm(embs, axis=1)
        
        for c in unique_classes:
            pos_mask = (labels == c)
            pos_norms = norms[pos_mask]
            neg_norms = norms[~pos_mask]
            
            if len(pos_norms) > 0 and len(neg_norms) > 0:
                u_stat, p_val = stats.mannwhitneyu(pos_norms, neg_norms, alternative='two-sided')
                results[f"class_{c}"] = {
                    "u_statistic": float(u_stat), 
                    "p_value": float(p_val)
                }
            
        return {"statistical_tests": results}

    @staticmethod
    def _compute_per_class_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """
        Calculates ROC AUC for each class using a One-vs-Rest (OvR) approach.
        """
        unique_classes = np.unique(y_true)
        roc_auc_per_class = {}

        for i, class_label in enumerate(unique_classes):
            # Create binary labels: 1= current class, 0=rest
            binary_labels = (y_true == class_label).astype(int)
            
            # Extract probabilities for the current class column
            class_probs = y_probs[:, i] # Assume sorted order
            
            try:
                score = roc_auc_score(binary_labels, class_probs)
                roc_auc_per_class[f"roc_auc_class_{class_label}"] = float(score)
            except ValueError:
                # Occurs if a class has no samples in the current batch/split
                roc_auc_per_class[f"roc_auc_class_{class_label}"] = 0.0
                
        return roc_auc_per_class
    
    def save_baseline_metrics(self, results: Dict[str, Any], path: str | Path):
        """
        Extracts the confusion matrix from results, normalizes it, and saves it.
        Only performs the save if the current space is EUCLIDEAN.
        """
        if self.space != Space.EUCLIDEAN:
            logger.info(f"Skipping baseline save: Current space is {self.space.name}, not EUCLIDEAN.")
            return

        if "confusion_matrix" not in results:
            logger.error("Confusion matrix not found in results dictionary.")
            return

        cm = results["confusion_matrix"]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path, cm)
        logger.success(f"Successfully saved normalized Euclidean baseline to {path}")