# run_scGAD.py
#################################################
# scGAD Function for Python Benchmarking Framework
# Input: Train/test AnnData objects and markers from rank_genes_groups
# Output: Standardized results format for CV framework
#################################################

import subprocess
import tempfile
import os
import pandas as pd
import anndata as ad
import numpy as np
import warnings
import pickle
from typing import Dict, List, Any


def run_scGAD_function(adata_train: ad.AnnData, adata_test: ad.AnnData, markers: pd.DataFrame) -> Dict[str, Any]:
    """
    scGAD Cell Type Annotation Function

    Purpose: Run scGAD algorithm using PyTorch-based semi-supervised open-set annotation
    Inputs:
      - adata_train: Training AnnData object (labeled reference / source)
      - adata_test: Test AnnData object to predict (unlabeled query / target)
      - markers: Marker genes dataframe (for interface consistency)
    Outputs: Dict with predictions, true_labels, confidence_scores, cell_ids
    Algorithm: Uses ZINB autoencoder with dual prototype classifiers (source/target),
               pairwise pseudo-labeling, cross-entropy + alignment + clustering losses.
               Supports open-set recognition (novel cell types in test data).
    """

    # Default return function for error handling
    def default_return():
        n_test_cells = len(adata_test.obs)
        return {
            'predictions': ['Unknown'] * n_test_cells,
            'true_labels': list(adata_test.obs['Ground_Truth_Celltype']),
            'confidence_scores': [0.0] * n_test_cells,
            'cell_ids': list(adata_test.obs.index)
        }

    # Validate input data
    if 'Ground_Truth_Celltype' not in adata_train.obs.columns:
        warnings.warn("Ground_Truth_Celltype not found in training data")
        return default_return()

    if 'Ground_Truth_Celltype' not in adata_test.obs.columns:
        warnings.warn("Ground_Truth_Celltype not found in test data")
        return default_return()

    # Check conda environment availability
    conda_env_path = "<WORKSTATION_HOME>/miniconda3/envs/scGAD_env"
    if not os.path.isdir(conda_env_path):
        warnings.warn(f"scGAD_env conda environment not found at {conda_env_path}. Please create it first.")
        return default_return()

    # Create temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Prepare file paths
            train_h5ad_path = os.path.join(temp_dir, "adata_train.h5ad")
            test_h5ad_path = os.path.join(temp_dir, "adata_test.h5ad")
            results_path = os.path.join(temp_dir, "results.pkl")

            # Get common genes between train and test
            common_genes = list(set(adata_train.var.index) & set(adata_test.var.index))
            if len(common_genes) < 1:
                warnings.warn("Too few common genes between train and test datasets")
                return default_return()

            # Subset both datasets to common genes and ensure consistent ordering
            adata_train_subset = adata_train[:, sorted(common_genes)].copy()
            adata_test_subset = adata_test[:, sorted(common_genes)].copy()

            # Save AnnData objects for inter-process communication
            adata_train_subset.write_h5ad(train_h5ad_path)
            adata_test_subset.write_h5ad(test_h5ad_path)

            # Determine project directory (where scGAD_package/ lives)
            project_dir = os.path.dirname(os.path.abspath(__file__))

            # Create Python script for scGAD execution
            scgad_script = f'''
import sys
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '{project_dir}')

try:
    import numpy as np
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    import time
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from itertools import cycle
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.cluster import KMeans
    from memory_profiler import memory_usage

    from scGAD_package.train_cross import scOpen1, extractor, cluster_match, AverageMeter, auxilarly_dis, cluster_acc, accuracy
    from scGAD_package.preprocess import normalize, annotation
    from scGAD_package.augmentation import mask_generator, pretext_generator
    from scGAD_package.layers import ZINBLoss

    # ---- Load data (NOT measured) ----
    adata_train = ad.read_h5ad('{train_h5ad_path}')
    adata_test = ad.read_h5ad('{test_h5ad_path}')

    # ---- Method execution (measured) ----
    _timing = [0.0]

    def _run_method():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _t0 = time.perf_counter()

        # --- Data Preparation ---
        n_train = adata_train.n_obs
        n_test = adata_test.n_obs

        # Extract data matrices
        if hasattr(adata_train.X, 'toarray'):
            X_train_raw = adata_train.X.toarray()
        else:
            X_train_raw = np.array(adata_train.X)
        if hasattr(adata_test.X, 'toarray'):
            X_test_raw = adata_test.X.toarray()
        else:
            X_test_raw = np.array(adata_test.X)

        X_train_raw = np.ceil(np.abs(X_train_raw)).astype(np.float32)
        X_test_raw = np.ceil(np.abs(X_test_raw)).astype(np.float32)

        cellname_train = np.array(adata_train.obs['Ground_Truth_Celltype'].astype(str))
        cellname_test = np.array(adata_test.obs['Ground_Truth_Celltype'].astype(str))

        # Determine seen (shared) and novel (test-only) classes
        train_types = set(cellname_train)
        test_types = set(cellname_test)
        seen_classes = sorted(train_types & test_types)
        novel_classes = sorted(test_types - train_types)
        all_classes = seen_classes + novel_classes

        if len(seen_classes) == 0:
            raise ValueError("No shared cell types between train and test")

        shared_classes = len(seen_classes)
        total_classes = len(all_classes)
        print(f"Seen classes: {{shared_classes}}, Novel classes: {{len(novel_classes)}}, Total: {{total_classes}}")

        # Create integer labels via dataset_labeling logic
        source_X_list, source_Y_list = [], []
        target_X_list, target_Y_list = [], []
        source_cn_list, target_cn_list = [], []

        for i, ct in enumerate(seen_classes):
            mask_s = cellname_train == ct
            source_X_list.append(X_train_raw[mask_s])
            source_cn_list.append(cellname_train[mask_s])
            source_Y_list.append(np.full(mask_s.sum(), i))
            mask_t = cellname_test == ct
            target_X_list.append(X_test_raw[mask_t])
            target_cn_list.append(cellname_test[mask_t])
            target_Y_list.append(np.full(mask_t.sum(), i))

        for j, ct in enumerate(novel_classes):
            mask_t = cellname_test == ct
            target_X_list.append(X_test_raw[mask_t])
            target_cn_list.append(cellname_test[mask_t])
            target_Y_list.append(np.full(mask_t.sum(), j + shared_classes))

        source_X = np.concatenate(source_X_list)
        source_cellname = np.concatenate(source_cn_list)
        source_Y = np.concatenate(source_Y_list).astype(int)
        source_batch_lbl = np.zeros_like(source_Y)

        target_X = np.concatenate(target_X_list)
        target_cellname = np.concatenate(target_cn_list)
        target_Y = np.concatenate(target_Y_list).astype(int)
        target_batch_lbl = np.ones_like(target_Y)

        # Concatenate for preprocessing
        X_full = np.concatenate([source_X, target_X], axis=0)
        Y_full = np.concatenate([source_Y, target_Y])
        batch_label_full = np.concatenate([source_batch_lbl, target_batch_lbl])
        cellname_full = np.concatenate([source_cellname, target_cellname])
        count_X = X_full.copy()

        # Preprocessing
        highly_genes = 2000
        if X_full.shape[1] <= highly_genes:
            highly_genes = None

        adata_full = sc.AnnData(X_full)
        adata_full.obs['batch'] = batch_label_full
        adata_full.obs['celltype'] = Y_full
        adata_full.obs['cellname'] = cellname_full
        adata_proc = normalize(adata_full, highly_genes=highly_genes,
                               size_factors=True, normalize_input=True, logtrans_input=True)

        X_proc = adata_proc.X.astype(np.float32)
        Y_proc = np.array(adata_proc.obs['celltype']).astype(int)
        cellname_proc = np.array(adata_proc.obs['cellname'])
        batch_label_proc = np.array(adata_proc.obs['batch']).astype(int)

        surviving_indices = np.array(adata_proc.obs.index, dtype=int)
        if highly_genes is not None:
            hvg_idx = np.array(adata_proc.var.index, dtype=int)
            count_X_proc = count_X[np.ix_(surviving_indices, hvg_idx)]
        else:
            gene_idx = np.array(adata_proc.var.index, dtype=int)
            count_X_proc = count_X[np.ix_(surviving_indices, gene_idx)]

        size_factor = np.array(adata_proc.obs.size_factors).reshape(-1, 1).astype(np.float32)
        batch_matrix = OneHotEncoder().fit_transform(batch_label_proc.reshape(-1, 1)).toarray()

        # Split back into source/target
        src_mask = batch_label_proc == 0
        tgt_mask = batch_label_proc == 1

        source_x = X_proc[src_mask]
        source_raw_x = count_X_proc[src_mask]
        source_y = Y_proc[src_mask]
        source_sf = size_factor[src_mask]
        source_batch = batch_matrix[src_mask]

        target_x = X_proc[tgt_mask]
        target_raw_x = count_X_proc[tgt_mask]
        target_y = Y_proc[tgt_mask]
        target_sf = size_factor[tgt_mask]
        target_batch = batch_matrix[tgt_mask]

        # Track which original test cells survived preprocessing
        surviving_test_positions = []
        original_n_source = len(source_X)
        for idx in surviving_indices:
            if idx >= original_n_source:
                surviving_test_positions.append(idx - original_n_source)

        num_batches = 2
        n_genes = X_proc.shape[1]
        print(f"After preprocessing: {{source_x.shape[0]}} source, {{target_x.shape[0]}} target, {{n_genes}} genes")

        # --- Model Setup ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"[scGAD] Running with GPU acceleration (device: {{device}}).", flush=True)
        else:
            print(f"[scGAD] Falling back to CPU. No GPU detected (device: {{device}}).", flush=True)

        random_seed = 8888
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        batch_size = 256
        if source_x.shape[0] < batch_size or target_x.shape[0] < batch_size:
            batch_size = min(source_x.shape[0], target_x.shape[0])

        # Structure=1: larger network
        model = scOpen1(n_genes, 128, shared_classes, total_classes, num_batches,
                        encodeLayer=[512, 256], decodeLayer=[256, 512],
                        activation="relu", tau=1.0)
        model = model.to(device)

        # Convert to tensors
        source_x_t = torch.tensor(source_x).to(device)
        source_raw_x_t = torch.tensor(source_raw_x).to(device)
        source_sf_t = torch.tensor(source_sf).to(device)
        source_y_t = torch.tensor(source_y).to(device)
        source_batch_t = torch.tensor(source_batch).float().to(device)
        target_x_t = torch.tensor(target_x).to(device)
        target_raw_x_t = torch.tensor(target_raw_x).to(device)
        target_sf_t = torch.tensor(target_sf).to(device)
        target_y_t = torch.tensor(target_y).to(device)
        target_batch_t = torch.tensor(target_batch).float().to(device)

        source_dataset = TensorDataset(source_x_t, source_raw_x_t, source_sf_t, source_y_t,
                                       torch.arange(source_x_t.shape[0]).to(device), source_batch_t)
        source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        target_dataset = TensorDataset(target_x_t, target_raw_x_t, target_sf_t, target_y_t,
                                       torch.arange(target_x_t.shape[0]).to(device), target_batch_t)
        target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_dataloader = DataLoader(
            TensorDataset(source_x_t, source_raw_x_t, source_sf_t, source_y_t,
                          torch.arange(source_x_t.shape[0]).to(device), source_batch_t),
            batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            TensorDataset(target_x_t, target_raw_x_t, target_sf_t, target_y_t,
                          torch.arange(target_x_t.shape[0]).to(device), target_batch_t),
            batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
        bce = nn.BCELoss()
        noi = 0.3
        top_k = 5
        quantile = 0.5

        epochs = 1500
        pretrain = 600
        midtrain = 1000
        interval = 10

        # Variables updated during eval
        train_targets = None
        test_label = None
        test_mask = None

        # --- Training Loop ---
        for epoch in range(epochs + 1):
            if epoch % interval == 0:
                train_embeddings, train_outputs, train_targets = extractor(model, train_dataloader, device)
                test_embeddings, test_outputs, test_targets = extractor(model, test_dataloader, device)

                if epoch <= pretrain:
                    kmeans_ = KMeans(n_clusters=total_classes, init="k-means++", random_state=random_seed).fit(
                        test_embeddings.cpu().numpy())
                    test_label = torch.from_numpy(kmeans_.labels_).to(device)
                    cluster_centers = torch.from_numpy(kmeans_.cluster_centers_).to(device)

                    test_mask = torch.zeros_like(test_label).float()
                    for i in range(total_classes):
                        if torch.sum(test_label == i) >= 2:
                            test_distance = torch.sum(torch.square(
                                test_embeddings[test_label == i].unsqueeze(1) - cluster_centers), dim=2)
                            intra_distance = test_distance[:, i]
                            inter_distance = torch.min(
                                test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[0]
                            test_score = -intra_distance / inter_distance
                            test_mask[test_label == i] = torch.where(
                                test_score >= torch.quantile(test_score, quantile),
                                torch.ones_like(test_score), torch.zeros_like(test_score))
                    test_label = test_label.to(train_targets.dtype)

                    label_centers = torch.zeros(shared_classes, cluster_centers.shape[1]).to(device)
                    for i in range(shared_classes):
                        if torch.sum(train_targets == i) > 0:
                            label_centers[i] = torch.mean(train_embeddings[train_targets == i], dim=0)
                    state_dict = model.state_dict()
                    state_dict['source_classifier.cluster'] = label_centers
                    state_dict['target_classifier.cluster'] = cluster_centers
                    model.load_state_dict(state_dict)
                else:
                    test_label = torch.max(test_outputs, dim=1)[1].to(train_targets.dtype)
                    test_mask = torch.zeros_like(test_label).float()
                    cluster_centers = model.target_classifier.cluster.data
                    for i in range(total_classes):
                        if torch.sum(test_label == i) >= 5:
                            test_distance = torch.sum(torch.square(
                                test_embeddings[test_label == i].unsqueeze(1) - cluster_centers), dim=2)
                            intra_distance = test_distance[:, i]
                            inter_distance = torch.min(
                                test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[0]
                            test_score = -intra_distance / inter_distance
                            test_mask[test_label == i] = torch.where(
                                test_score >= torch.quantile(test_score, quantile),
                                torch.ones_like(test_score), torch.zeros_like(test_score))

                if epoch % 100 == 0:
                    print(f"Epoch {{epoch}}/{{epochs}}")

            # Training step
            source_dataloader_iter = cycle(source_dataloader)
            model.train()

            for batch_idx, (x_t, raw_x_t, sf_t, y_t, index_t, batch_t) in enumerate(target_dataloader):
                (x_s, raw_x_s, sf_s, y_s, index_s, batch_s) = next(source_dataloader_iter)

                labeled_len = len(y_s)
                x_all = torch.cat([x_s, x_t], dim=0)
                mask_all_ = mask_generator(noi, x_all.cpu().numpy())
                m_new, x_tilde = pretext_generator(mask_all_, x_all.cpu().numpy())
                m_new = torch.from_numpy(m_new).to(torch.float32).to(device)
                x_tilde = torch.from_numpy(x_tilde).to(torch.float32).to(device)

                z_s1, mean_s1, disp_s1, pi_s1, mask_s1, output_ss1, output_st1 = model(x_all[:labeled_len], batch_s)
                z_s2, mean_s2, disp_s2, pi_s2, mask_s2, output_ss2, output_st2 = model(x_tilde[:labeled_len], batch_s)
                z_t1, mean_t1, disp_t1, pi_t1, mask_t1, output_ts1, output_tt1 = model(x_all[labeled_len:], batch_t)
                z_t2, mean_t2, disp_t2, pi_t2, mask_t2, output_ts2, output_tt2 = model(x_tilde[labeled_len:], batch_t)

                recon_loss = ZINBLoss().to(device)(x=raw_x_s, mean=mean_s1, disp=disp_s1, pi=pi_s1, scale_factor=sf_s) \\
                             + ZINBLoss().to(device)(x=raw_x_t, mean=mean_t1, disp=disp_t1, pi=pi_t1, scale_factor=sf_t)
                recon_loss = recon_loss - torch.mean(
                    m_new[:labeled_len] * torch.log(mask_s2 + 1e-8) + (1. - m_new[:labeled_len]) * torch.log(1. - mask_s2 + 1e-8)) \\
                             - torch.mean(
                    m_new[labeled_len:] * torch.log(mask_t2 + 1e-8) + (1. - m_new[labeled_len:]) * torch.log(1. - mask_t2 + 1e-8))

                feat1 = torch.cat([z_s1, z_t1], dim=0)
                feat2 = torch.cat([z_s2, z_t2], dim=0)
                output1 = torch.cat([output_st1, output_tt1], dim=0)
                output2 = torch.cat([output_st2, output_tt2], dim=0)
                prob1 = output1 / torch.sum(output1, dim=1, keepdim=True)
                prob2 = output2 / torch.sum(output2, dim=1, keepdim=True)
                _, pred1 = prob1.max(1)
                _, pred2 = prob2.max(1)

                label_gt = (y_s.view(-1, 1) - y_s.view(1, -1)).float()
                label_gt = torch.where(label_gt == 0., torch.ones_like(label_gt), torch.zeros_like(label_gt))

                with torch.no_grad():
                    pd1 = pred1.detach()
                    pd2 = pred2.detach()
                    ps1 = (pd1.view(-1, 1) - pd1.view(1, -1)).float()
                    ps1 = torch.where(ps1 == 0., torch.ones_like(ps1), torch.zeros_like(ps1))
                    ps2 = (pd2.view(-1, 1) - pd2.view(1, -1)).float()
                    ps2 = torch.where(ps2 == 0., torch.ones_like(ps2), torch.zeros_like(ps2))

                    fd1 = feat1.detach()
                    fs1 = torch.zeros_like(ps1).to(device)
                    d1 = torch.sum(torch.square(fd1.unsqueeze(1) - fd1), dim=2)
                    _, near1 = torch.topk(d1, dim=-1, largest=False, k=top_k + 1)
                    near1 = near1[:, 1:]
                    fn1 = fd1[near1]
                    d1_ = torch.sum(torch.square(fn1.unsqueeze(2) - fd1), dim=3)
                    _, near1_ = torch.topk(d1_, dim=-1, largest=False, k=top_k + 1)
                    near1_ = near1_[:, :, 1:]
                    idx1 = torch.arange(fd1.shape[0]).unsqueeze(-1).unsqueeze(-1).to(device)
                    m1 = (near1_ == idx1).sum(-1).float()
                    fs1 = fs1.scatter_(1, near1, m1)

                    fd2 = feat2.detach()
                    fs2 = torch.zeros_like(ps2).to(device)
                    d2 = torch.sum(torch.square(fd2.unsqueeze(1) - fd2), dim=2)
                    _, near2 = torch.topk(d2, dim=-1, largest=False, k=top_k + 1)
                    near2 = near2[:, 1:]
                    fn2 = fd2[near2]
                    d2_ = torch.sum(torch.square(fn2.unsqueeze(2) - fd2), dim=3)
                    _, near2_ = torch.topk(d2_, dim=-1, largest=False, k=top_k + 1)
                    near2_ = near2_[:, :, 1:]
                    idx2 = torch.arange(fd2.shape[0]).unsqueeze(-1).unsqueeze(-1).to(device)
                    m2 = (near2_ == idx2).sum(-1).float()
                    fs2 = fs2.scatter_(1, near2, m2)

                    fs1 = fs1 * ps1
                    fs1[:labeled_len, :labeled_len] = label_gt.clone()
                    fs2 = fs2 * ps2
                    fs2[:labeled_len, :labeled_len] = label_gt.clone()

                unlabel_sim1 = torch.mm(prob1, prob1.t())
                unlabel_sim2 = torch.mm(prob2, prob2.t())
                unlabel_sim3 = torch.mm(prob1, prob2.t())
                pse_loss = torch.mean(torch.sum(-fs1 * torch.log(torch.sigmoid(unlabel_sim1) + 1e-8), dim=1)) \\
                           + torch.mean(torch.sum(-fs2 * torch.log(torch.sigmoid(unlabel_sim2) + 1e-8), dim=1)) \\
                           + torch.mean(torch.sum(-fs1 * fs2 * torch.log(torch.sigmoid(unlabel_sim3) + 1e-8), dim=1))

                labeled_onehot = F.one_hot(train_targets, num_classes=shared_classes)[index_s.long()].float()
                unlabeled_onehot = F.one_hot(test_label, num_classes=total_classes)[index_t.long()].float()
                unlabeled_mask_batch = test_mask[index_t.long()].float()

                oss1 = output_ss1 / torch.sum(output_ss1, dim=1, keepdim=True)
                oss2 = output_ss2 / torch.sum(output_ss2, dim=1, keepdim=True)
                ost1 = output_st1 / torch.sum(output_st1, dim=1, keepdim=True)
                ost2 = output_st2 / torch.sum(output_st2, dim=1, keepdim=True)
                ott1 = output_tt1 / torch.sum(output_tt1, dim=1, keepdim=True)
                ott2 = output_tt2 / torch.sum(output_tt2, dim=1, keepdim=True)

                ce_loss = -torch.mean(torch.sum(labeled_onehot * torch.log(torch.clamp(oss1, min=1e-8)), dim=1)) \\
                          - torch.mean(torch.sum(labeled_onehot * torch.log(torch.clamp(oss2, min=1e-8)), dim=1)) \\
                          - torch.mean(unlabeled_mask_batch * torch.sum(unlabeled_onehot * torch.log(torch.clamp(ott1, min=1e-8)), dim=1)) \\
                          - torch.mean(unlabeled_mask_batch * torch.sum(unlabeled_onehot * torch.log(torch.clamp(ott2, min=1e-8)), dim=1))

                align_loss = torch.mean(torch.sum(-auxilarly_dis(ost1) * torch.log(torch.clamp(ost1, min=1e-8)), dim=1)) \\
                             + torch.mean((1.0 - unlabeled_mask_batch) * torch.sum(-auxilarly_dis(ott1) * torch.log(torch.clamp(ott1, min=1e-8)), dim=1)) \\
                             + torch.mean(torch.sum(-auxilarly_dis(ost2) * torch.log(torch.clamp(ost2, min=1e-8)), dim=1)) \\
                             + torch.mean((1.0 - unlabeled_mask_batch) * torch.sum(-auxilarly_dis(ott2) * torch.log(torch.clamp(ott2, min=1e-8)), dim=1))

                ls1 = torch.mm(ost1, ost1.t())
                ls2 = torch.mm(ost2, ost2.t())
                ls3 = torch.mm(ost1, ost2.t())
                clu_loss = bce(ls1.view(-1, 1), label_gt.view(-1, 1)) + bce(ls2.view(-1, 1), label_gt.view(-1, 1)) \\
                           + bce(ls3.view(-1, 1), label_gt.view(-1, 1))

                if epoch < pretrain:
                    loss = recon_loss
                else:
                    loss = recon_loss + ce_loss + align_loss + pse_loss + clu_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # --- Final Prediction Extraction ---
        print("Extracting final predictions...")
        model.eval()
        all_preds = []
        all_confs = []
        with torch.no_grad():
            for _, data in enumerate(test_dataloader):
                x_t_b, _, _, _, _, batch_t_b = data
                z, _, _, _, _, out_s, out_t = model(x_t_b, batch_t_b)
                out_c = torch.cat([out_s, out_t], dim=1)
                conf_c, pred_c = out_c.max(1)
                all_preds.append(pred_c.cpu().numpy())
                all_confs.append(conf_c.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_confs = np.concatenate(all_confs)

        # Apply cluster mapping using a lookup to avoid sequential collision
        label_centers = model.source_classifier.cluster.data
        cluster_centers = model.target_classifier.cluster.data
        cluster_mappings = cluster_match(label_centers, cluster_centers)

        # Build a full mapping array: for combined output indices → final class index
        # Combined output layout: [source_0..source_(S-1), target_0..target_(T-1)]
        # where S=shared_classes, T=total_classes
        combined_size = shared_classes + total_classes
        final_map = np.arange(combined_size)  # default: identity
        # Source outputs (0..S-1) map directly to seen class indices
        for i in range(shared_classes):
            final_map[i] = i
        # Target outputs (S..S+T-1): map matched clusters to seen classes, rest keep as-is
        for i in range(shared_classes):
            final_map[shared_classes + cluster_mappings[i]] = i
        # Novel target clusters (unmatched) keep their target index

        # Convert integer predictions to cell type names
        pred_celltype_names = []
        for p in all_preds:
            mapped_idx = final_map[int(p)] if int(p) < combined_size else int(p)
            if mapped_idx < len(all_classes):
                pred_celltype_names.append(all_classes[int(mapped_idx)])
            else:
                pred_celltype_names.append("Unknown")

        # Map back to original test cell ordering
        predictions = ["Unknown"] * n_test
        confidence_scores = [0.0] * n_test
        true_labels = list(cellname_test)
        cell_ids = list(adata_test.obs.index)

        # surviving_test_positions maps processed test cells back to original target_X positions
        # target_X was built from seen_classes first, then novel_classes
        # We need a reverse mapping from target_X position to original test cell position
        target_to_original = []
        for ct in seen_classes:
            idxs = np.where(cellname_test == ct)[0]
            target_to_original.extend(idxs.tolist())
        for ct in novel_classes:
            idxs = np.where(cellname_test == ct)[0]
            target_to_original.extend(idxs.tolist())

        for i, pos in enumerate(surviving_test_positions):
            if pos < len(target_to_original) and i < len(pred_celltype_names):
                orig_idx = target_to_original[pos]
                if orig_idx < n_test:
                    predictions[orig_idx] = pred_celltype_names[i]
                    confidence_scores[orig_idx] = float(all_confs[i])

        # Normalize confidence scores to [0, 1]
        if max(confidence_scores) > 0:
            max_conf = max(c for c in confidence_scores if c > 0)
            confidence_scores = [c / max_conf if c > 0 else 0.0 for c in confidence_scores]

        _timing[0] = time.perf_counter() - _t0
        _peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        _peak_vram_mb = _peak_vram_bytes / (1024 * 1024)
        return predictions, confidence_scores, true_labels, cell_ids, _peak_vram_mb

    mem_list, (predictions, confidence_scores, true_labels, cell_ids, peak_vram_mb) = memory_usage(
        (_run_method, [], {{}}), interval=0.1, include_children=True, retval=True)
    peak_system_memory_mb = max(mem_list)
    method_walltime = _timing[0]

    results = {{
        'predictions': list(predictions),
        'confidence_scores': confidence_scores,
        'true_labels': true_labels,
        'cell_ids': cell_ids,
        'peak_system_memory_mb': peak_system_memory_mb,
        'peak_vram_mb': peak_vram_mb,
        'method_walltime': method_walltime,
    }}

    print(f"Peak system memory: {{peak_system_memory_mb:.2f}} MB, VRAM: {{peak_vram_mb:.2f}} MB, walltime: {{method_walltime:.2f}}s")

    with open('{results_path}', 'wb') as f:
        pickle.dump(results, f)

    print(f"Predictions: {{len(predictions)}} cells, unique: {{set(predictions)}}")

except Exception as e:
    print(f"scGAD execution failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

            # Execute scGAD in conda environment
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'

            result = subprocess.run(
                [
                    "<WORKSTATION_HOME>/miniconda3/condabin/conda",
                    "run",
                    "-n", "scGAD_env",
                    "python",
                    "-c", scgad_script
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour timeout
            )

            if result.returncode != 0:
                warnings.warn(f"scGAD execution failed: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return default_return()

            # Print subprocess output
            print(result.stdout)

            # Load results from pickle file
            if not os.path.exists(results_path):
                warnings.warn("scGAD results file not found")
                return default_return()

            with open(results_path, 'rb') as f:
                results = pickle.load(f)

            cell_predictions = results['predictions']
            confidence_scores = results['confidence_scores']
            true_labels = results['true_labels']
            cell_ids = results['cell_ids']

            # Ensure predictions have correct length
            if len(cell_predictions) != len(adata_test.obs):
                warnings.warn(f"Prediction length mismatch. Expected: {len(adata_test.obs)}, Got: {len(cell_predictions)}")
                return default_return()

            # Convert predictions to strings and handle missing values
            cell_predictions = [str(pred) if pd.notna(pred) else "Unknown" for pred in cell_predictions]

            # Process confidence scores
            confidence_scores = [
                0.0 if pred == "Unknown" else (max(0.0, min(1.0, float(conf))) if pd.notna(conf) and isinstance(conf, (int, float)) else 0.5)
                for pred, conf in zip(cell_predictions, confidence_scores)
            ]

            print(f"scGAD completed successfully:")
            print(f"  - Total predictions: {len(cell_predictions)}")
            print(f"  - Unknown predictions: {sum(1 for p in cell_predictions if p == 'Unknown')}")
            print(f"  - Unique predictions: {len(set(cell_predictions))}")

            # Return standardized format
            return {
                'predictions': cell_predictions,
                'true_labels': true_labels,
                'confidence_scores': confidence_scores,
                'cell_ids': cell_ids,
                'peak_memory_mb':  results.get('peak_system_memory_mb', None),
                'peak_vram_mb':    results.get('peak_vram_mb', None),
                'method_walltime': results.get('method_walltime', None),
            }

        except subprocess.TimeoutExpired:
            warnings.warn("scGAD execution timed out after 24 hours")
            return default_return()
        except Exception as e:
            warnings.warn(f"scGAD error: {str(e)}")
            return default_return()


# For backward compatibility
run_scGAD = run_scGAD_function
