import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm

import torch
import numpy as np
from utils import encode_prompt
from metrics import calculate_metric
from torch.nn import functional as F

logger = logging.getLogger(__name__)

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

@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]