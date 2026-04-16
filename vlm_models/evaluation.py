import os
import logging
import torch
import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_DIR = "/workspace/wangfei154/datasets/" # None

def preprocess_logits_for_metrics(logits, labels):
    """
    Applies argmax on logits batch-by-batch to prevent OOM during evaluation.
    Reduces memory footprint from [batch, seq_len, vocab_size] to [batch, seq_len].
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_token_metrics(eval_pred):
    """
    Computes token-level accuracy (mainly for Trainer's internal progress logging).
    """
    predictions, labels = eval_pred
    # Causal LM Shift
    preds_shifted = predictions[:, :-1]
    labels_shifted = labels[:, 1:]
    
    mask = labels_shifted != -100
    valid_preds = preds_shifted[mask]
    valid_labels = labels_shifted[mask]
    
    if len(valid_preds) == 0:
        return {"accuracy": 0.0}
        
    accuracy = (valid_preds == valid_labels).mean()
    return {"accuracy": float(accuracy)}

def evaluate_vlm_predictions(predict_results, dataset_name, metrics_dict, trainer_logger=None):
    """
    Task-agnostic evaluation router. 
    Routes to subset-aware evaluation for ScienceQA, or generic strict-match for others.
    """
    log = trainer_logger or logger
    predictions = predict_results.predictions
    labels = predict_results.label_ids
    
    preds_shifted = predictions[:, :-1]
    labels_shifted = labels[:, 1:]
    mask = labels_shifted != -100
    
    dataset_name = dataset_name.lower()
    
    if "scienceqa" in dataset_name:
        log.info("ScienceQA detected. Running subset-aware evaluation...")
        raw_eval_data = load_dataset(os.path.join(DATASET_DIR, "derek-thomas/ScienceQA"), split="validation")
        
        subset_correct = {"NAT": 0, "SOC": 0, "LAN": 0, "TXT": 0, "IMG": 0, "NO": 0, "G1-6": 0, "G7-12": 0}
        subset_total   = {"NAT": 0, "SOC": 0, "LAN": 0, "TXT": 0, "IMG": 0, "NO": 0, "G1-6": 0, "G7-12": 0}
        global_correct, global_total = 0, 0
        
        for i in range(len(preds_shifted)):
            sample_pred = preds_shifted[i]
            sample_label = labels_shifted[i]
            sample_mask = mask[i]
            
            if not sample_mask.any(): continue
                
            # is_correct = (sample_pred[sample_mask] == sample_label[sample_mask]).all().item()
            is_correct = (sample_pred[sample_mask][0] == sample_label[sample_mask][0]).item()
            global_total += 1
            if is_correct: global_correct += 1
            
            item = raw_eval_data[i]
            subject = item.get("subject", "")
            hint = item.get("hint", "")
            image = item.get("image", None)
            grade = item.get("grade", "")
            
            if subject == "natural science": subset_total["NAT"] += 1; subset_correct["NAT"] += int(is_correct)
            elif subject == "social science": subset_total["SOC"] += 1; subset_correct["SOC"] += int(is_correct)
            elif subject == "language science": subset_total["LAN"] += 1; subset_correct["LAN"] += int(is_correct)
                
            has_txt = bool(hint and hint.strip())
            has_img = image is not None
            if has_img: subset_total["IMG"] += 1; subset_correct["IMG"] += int(is_correct)
            if has_txt: subset_total["TXT"] += 1; subset_correct["TXT"] += int(is_correct)
            if not has_img and not has_txt: subset_total["NO"] += 1; subset_correct["NO"] += int(is_correct)
                
            if grade in ["grade1", "grade2", "grade3", "grade4", "grade5", "grade6"]: subset_total["G1-6"] += 1; subset_correct["G1-6"] += int(is_correct)
            elif grade in ["grade7", "grade8", "grade9", "grade10", "grade11", "grade12"]: subset_total["G7-12"] += 1; subset_correct["G7-12"] += int(is_correct)
                
        log.info("================ ScienceQA Subset Accuracy ================")
        for subset in subset_total:
            if subset_total[subset] > 0:
                acc = subset_correct[subset] / subset_total[subset]
                metrics_dict[f"eval_acc_{subset}"] = acc
                log.info(f"Subset {subset:<5} | Acc: {acc*100:.2f}% ({subset_correct[subset]}/{subset_total[subset]})")
        
        instance_acc = global_correct / global_total if global_total > 0 else 0
        metrics_dict["eval_instance_accuracy"] = instance_acc
        log.info("-" * 59)
        log.info(f"Global Instance Acc | {instance_acc*100:.2f}%")
        log.info("===========================================================")

    else:
        log.info(f"Generic evaluation fallback for dataset: {dataset_name}")
        global_correct, global_total = 0, 0
        
        for i in range(len(preds_shifted)):
            sample_pred = preds_shifted[i]
            sample_label = labels_shifted[i]
            sample_mask = mask[i]
            
            if not sample_mask.any(): continue
                
            is_correct = (sample_pred[sample_mask] == sample_label[sample_mask]).all().item()
            global_total += 1
            if is_correct: global_correct += 1
            
        instance_acc = global_correct / global_total if global_total > 0 else 0
        metrics_dict["eval_instance_accuracy"] = instance_acc
        log.info("================ Generic Evaluation ================")
        log.info(f"Global Instance Acc | {instance_acc*100:.2f}% ({global_correct}/{global_total})")
        log.info("====================================================")
        
    return metrics_dict