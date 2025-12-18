import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from utils import encode_prompt
from metrics import calculate_metric

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


class Evaluator:
    """
    Helper class to replicate the exact evaluation logic from the original ZO codebase.
    It handles prompt encoding, candidate ranking, and metric calculation.
    """
    def __init__(self, args, task, tokenizer, model):
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.model = model

    def inference_forward(self, input_ids, option_len=None, generation=False):
        """
        Forward pass for inference. 
        Calculates log-likelihood of the option part or generates text.
        """
        # Ensure input is on the correct device
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            attention_mask = torch.ones_like(input_ids)
            # Autoregressive generation (e.g., for SQuAD, DROP)
            outputs = self.model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                pad_token_id=self.tokenizer.pad_token_id, 
                do_sample=self.args.sampling, 
                temperature=self.args.temperature, 
                num_beams=self.args.num_beams, 
                top_p=self.args.top_p, 
                top_k=self.args.top_k, 
                max_new_tokens=min(self.args.max_new_tokens, self.args.max_length - input_ids.size(1)), 
                num_return_sequences=1, 
                eos_token_id=[self.tokenizer.encode(self.args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # Decode generated tokens
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            # Classification/Multiple-Choice: Calculate log-probabilities
            with torch.inference_mode():
                # Use the underlying model directly to avoid ZO-specific forward wrappers if applied
                if hasattr(self.model, "original_forward"):
                    outputs = self.model.original_forward(input_ids=input_ids)
                else:
                    outputs = self.model(input_ids=input_ids)
                
            logits = outputs.logits
            
            # Shift logits and labels for autoregressive loss
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            # Gather probabilities of the ground truth tokens
            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            
            # Return only the log-probs corresponding to the option/answer part
            return selected_log_probs[-option_len:]

    def one_step_pred(self, eval_sample, train_samples, verbose=False):
        """
        Perform prediction on a single evaluation sample.
        Supports both Generation tasks and Classification/Multiple-Choice tasks.
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        # Encode prompts (Standard)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(self.args.template_ver),
            train_samples, eval_sample, self.tokenizer, 
            max_length=self.args.max_length, 
            generation=self.task.generation, 
            max_new_tokens=self.args.max_new_tokens
        )

        # SFC Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task, self.task.get_template(template_version=self.args.template_ver), 
                train_samples, eval_sample, self.tokenizer, 
                max_length=self.args.max_length, 
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, 
                generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        # --- Prediction Logic ---
        if self.task.generation:
            # Generation Task
            output_text = self.inference_forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # Ranking Task (Classification)
            outputs = []
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                # Calculate score for this candidate
                selected_log_probs = self.inference_forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                sfc_log_probs = None
                
                if self.args.sfc or self.args.icl_sfc:
                    sfc_log_probs = self.inference_forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_log_probs})

            # Calculate calibration scores
            if self.args.sfc or self.args.icl_sfc:
                # Calibrated: log P(cand|input) - log P(cand|empty)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # Default: Average Log-Likelihood (Length Normalized)
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            # Determine predicted ID
            predicted_id = int(np.argmax(scores))
            
            # Map correct candidate text to index/indices
            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=predicted_id)

    def evaluate(self, eval_samples, train_samples, one_train_set_per_eval_sample=False, verbose_len=0):
        """
        Run evaluation loop over the validation/test set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"Evaluation: {len(eval_samples)} samples (One train set per eval sample).")
        else:
            logger.info(f"Evaluation: {len(train_samples)} training samples, {len(eval_samples)} validation samples.")

        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(eval_sample, train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, verbose=(eval_id < verbose_len-1))
            )
        
        # Calculate final metric (Accuracy, F1, etc.)
        metric_name = getattr(self.task, "metric_name", "accuracy")
        score = calculate_metric(predictions, metric_name)
        
        return {metric_name: score}


class BatchedEvaluator:
    """
    Evaluator that supports batch processing to accelerate the inference process.
    Replaces the original Evaluator to support batch inference.
    """
    def __init__(self, args, task, tokenizer, model):
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.model = model

    def _collate_fn(self, batch):
        """
        Pads input_ids and converts them to Tensors.
        """
        # Find the maximum length in the current batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids_padded = []
        attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for item in batch:
            input_ids = item['input_ids']
            # Left Padding or Right Padding depends on the model configuration.
            # Generally, generation tasks prefer Left Padding, while classification can use Right Padding.
            # Since encode_prompt handles truncation, we perform simple padding here.
            # Note: If it's a generation task, HF 'generate' usually requires Left Padding; 
            # check 'tokenizer.padding_side'.
            
            padding_len = max_len - len(input_ids)
            if self.tokenizer.padding_side == 'left':
                 padded_ids = [pad_token_id] * padding_len + input_ids
                 mask = [0] * padding_len + [1] * len(input_ids)
            else:
                 padded_ids = input_ids + [pad_token_id] * padding_len
                 mask = [1] * len(input_ids) + [0] * padding_len
            
            input_ids_padded.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "metadata": [item['metadata'] for item in batch],  # Keep original metadata for restoration
            "option_len": [item['option_len'] for item in batch]
        }

    def batch_inference(self, dataloader, generation=False):
        """
        Execute batch inference.
        """
        results = []
        
        for batch in tqdm(dataloader, desc="Batch Inference"):
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            option_lens = batch['option_len']
            metadata = batch['metadata']

            if generation:
                # Generation Task
                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=self.args.sampling,
                        temperature=self.args.temperature,
                        max_new_tokens=self.args.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # Decode outputs
                for i, output in enumerate(outputs):
                    # Remove the input part, keep only the newly generated content
                    input_len = input_ids.shape[1]
                    if self.tokenizer.padding_side == 'left':
                         # Left padding means input_len includes padding; 
                         # generated output usually contains the full sequence.
                         generated_text = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                    else:
                         generated_text = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                    
                    results.append({
                        "metadata": metadata[i],
                        "output": generated_text
                    })

            else:
                # Classification/Ranking Task - Calculate Log Probability
                with torch.inference_mode():
                    # Get Logits
                    if hasattr(self.model, "original_forward"):
                        outputs = self.model.original_forward(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                logits = outputs.logits # (B, L, V)
                
                # Calculate Loss / Log Probs
                # Shift: Logits[t] predicts Labels[t+1]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                log_probs = F.log_softmax(shift_logits, dim=-1) # (B, L-1, V)
                
                # Gather log_prob of the correct token
                # Use shift_labels as indices
                gathered_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1) # (B, L-1)
                
                # Move back to CPU for processing
                gathered_log_probs = gathered_log_probs.cpu()
                
                for i in range(len(metadata)):
                    # Extract the part corresponding to the option
                    opt_len = option_lens[i]
                    if opt_len == 0:
                        score = 0.0
                    else:
                        # Get log prob of the last `opt_len` tokens
                        # Note: gathered_log_probs length is L-1, corresponding to input_ids[1:]
                        # The option is at the very end of the input
                        score_tokens = gathered_log_probs[i, -opt_len:]
                        score = score_tokens.mean().item() # Use mean LogLikelihood
                        # If Sum is needed, use .sum().item()
                    
                    results.append({
                        "metadata": metadata[i], # Contains sample_id, candidate_id
                        "score": score
                    })
                    
        return results

    def evaluate(self, eval_samples, train_samples, one_train_set_per_eval_sample=False, batch_size=None):
        """
        Main evaluation function: Preprocess data -> Build Batch -> Inference -> Aggregate results.
        """
        if batch_size is None:
            batch_size = self.args.per_device_eval_batch_size if hasattr(self.args, 'per_device_eval_batch_size') else 16

        logger.info(f"Starting Batch Evaluation with Batch Size {batch_size}...")
        
        # 1. Preprocess all samples (Flatten)
        flat_data = [] # Store each inference request (One Candidate of One Sample)
        
        # Determine mode
        generation_mode = self.task.generation
        
        logger.info("Encoding prompts...")
        for eval_id, eval_sample in enumerate(tqdm(eval_samples, desc="Preparing Data")):
            # Get corresponding ICL Demonstrations
            curr_train_samples = train_samples[eval_id] if one_train_set_per_eval_sample else train_samples
            
            # Encode (call utils.py's encode_prompt)
            # encoded_candidates: List[List[int]], option_lens: List[int]
            encoded_candidates, option_lens = encode_prompt(
                self.task, self.task.get_template(self.args.template_ver),
                curr_train_samples, eval_sample, self.tokenizer, 
                max_length=self.args.max_length, 
                generation=generation_mode, 
                max_new_tokens=self.args.max_new_tokens
            )
            
            # Construct data item
            if generation_mode:
                # Generation task has only one Input
                flat_data.append({
                    "input_ids": encoded_candidates[0],
                    "option_len": 0, # Generation task doesn't need option log prob
                    "metadata": {"sample_id": eval_id, "type": "std"}
                })
            else:
                # Ranking Task: Treat each Candidate as an independent Batch Item
                for cand_id, (enc_ids, opt_len) in enumerate(zip(encoded_candidates, option_lens)):
                    flat_data.append({
                        "input_ids": enc_ids,
                        "option_len": opt_len,
                        "metadata": {"sample_id": eval_id, "cand_id": cand_id, "type": "std"}
                    })
            
            # Handle SFC (Surface Form Competition) - if enabled
            if (self.args.sfc or self.args.icl_sfc) and not generation_mode:
                 sfc_encoded, sfc_lens = encode_prompt(
                    self.task, self.task.get_template(self.args.template_ver), 
                    curr_train_samples, eval_sample, self.tokenizer, 
                    max_length=self.args.max_length, 
                    sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, 
                    generation=generation_mode
                )
                 for cand_id, (enc_ids, opt_len) in enumerate(zip(sfc_encoded, sfc_lens)):
                    flat_data.append({
                        "input_ids": enc_ids,
                        "option_len": opt_len,
                        "metadata": {"sample_id": eval_id, "cand_id": cand_id, "type": "sfc"}
                    })

        # 2. Build DataLoader
        dataloader = DataLoader(
            flat_data, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._collate_fn,
            num_workers=4 # Adjust based on CPU core count
        )
        
        # 3. Execute Batch Inference
        raw_results = self.batch_inference(dataloader, generation=generation_mode)
        
        # 4. Aggregate results (Group by Sample ID)
        grouped_results = {} # sample_id -> { "std": {cand_id: score}, "sfc": {cand_id: score}, "output": str }
        
        for res in raw_results:
            meta = res['metadata']
            sid = meta['sample_id']
            if sid not in grouped_results:
                grouped_results[sid] = {"std": {}, "sfc": {}, "output": None}
            
            if generation_mode:
                grouped_results[sid]["output"] = res['output']
            else:
                rtype = meta['type'] # std or sfc
                cid = meta['cand_id']
                grouped_results[sid][rtype][cid] = res['score']
                
        # 5. Calculate final Metrics
        final_predictions = []
        
        for eval_id, eval_sample in enumerate(eval_samples):
            res = grouped_results.get(eval_id)
            if not res:
                logger.error(f"Sample {eval_id} missing from results.")
                continue
                
            if generation_mode:
                # Generation task directly takes the Output
                pred_obj = Prediction(
                    correct_candidate=eval_sample.correct_candidate, 
                    predicted_candidate=res['output']
                )
            else:
                # Ranking Task: Calculate Argmax
                scores_std = [res["std"][i] for i in range(len(eval_sample.candidates))]
                
                final_scores = scores_std
                # Apply SFC calibration
                if self.args.sfc or self.args.icl_sfc:
                    scores_sfc = [res["sfc"][i] for i in range(len(eval_sample.candidates))]
                    # Calibrated Score: P(cand|context) - P(cand|empty) (Use subtraction for log probs)
                    # Alternatively, could use division (P / P_sfc)
                    final_scores = [s - c for s, c in zip(scores_std, scores_sfc)]
                
                predicted_id = int(np.argmax(final_scores))
                
                # Handle correct answer (can be List or Int)
                if isinstance(eval_sample.correct_candidate, list):
                    correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
                else:
                    correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

                pred_obj = Prediction(
                    correct_candidate=correct_candidate_id, 
                    predicted_candidate=predicted_id
                )
                
            final_predictions.append(pred_obj)

        # Calculate Metric
        metric_name = getattr(self.task, "metric_name", "accuracy")
        score = calculate_metric(final_predictions, metric_name)
        
        return {metric_name: score}