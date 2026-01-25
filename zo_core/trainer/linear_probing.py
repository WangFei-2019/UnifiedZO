import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm
import logging
import numpy as np

logger = logging.getLogger(__name__)

def perform_linear_probing(args, model, tokenizer, train_dataset, collator):
    """
    Executes Linear Probing: Extract features -> Train Logistic Regression -> Update LM Head.
    Independent module to keep the main Trainer clean.
    """
    logger.info("Starting Linear Probing...")

    # 1. Helper to retrieve the LM Head (adapts to different model architectures)
    def _get_token_prediction_layer(model):
        # Support OPT, LLaMA, Mistral, etc.
        if "opt" in args.model_name.lower():
            return model.lm_head
        elif "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
            return model.lm_head
        else:
            # Fallback
            if hasattr(model, 'lm_head'):
                return model.lm_head
            raise NotImplementedError(f"Model type for {args.model_name} not explicitly supported in Linear Probing helper.")

    # 2. Register Forward Hook to extract features (Pre-softmax)
    captured_feats = {} 
    def __hook(model_, input_, output_):
        # input_[0] is usually the hidden states before the LM Head
        captured_feats["features"] = input_[0].detach()

    prediction_layer = _get_token_prediction_layer(model)
    handle = prediction_layer.register_forward_hook(__hook)

    # 3. Prepare DataLoader (Sequential Sampling for feature extraction)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        collate_fn=collator,
        shuffle=False 
    )

    targets = []
    features = []

    logger.info("Extracting features from training set...")
    model.eval()
    
    with torch.inference_mode():
        for step, inputs in enumerate(tqdm(train_dataloader, desc="LP Feature Extraction")):
            # Move inputs to device
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Forward pass triggers Hook (no loss calculation needed)
            outputs = model(**inputs)
            
            feature = captured_feats["features"] # (batch, seq_len, hidden_dim)
            target = inputs["labels"]            # (batch, seq_len)

            # --- Alignment logic (consistent with MeZO) ---
            # For Causal LM, feature at t predicts target at t+1
            feature = feature[:, :-1, :]
            target = target[:, 1:]
            
            # Extract token features corresponding to the Option (Answer) only
            if "option_len" in inputs:
                for _i, _len in enumerate(inputs["option_len"]):
                    # Extract the last `_len` tokens
                    features.append(feature[_i, -_len:, :].cpu())
                    targets.append(target[_i, -_len:].cpu())
            else:
                # Fallback: extract all tokens (usually not recommended for this task)
                features.append(feature.reshape(-1, feature.shape[-1]).cpu())
                targets.append(target.reshape(-1).cpu())

    handle.remove() # Remove Hook

    # 4. Prepare data for Sklearn
    features = torch.cat(features, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    
    logger.info(f"Feature shape: {features.shape}, Target shape: {targets.shape}")

    # 5. Configure and train Logistic Regression
    use_bias = False 
    if "bert" in args.model_name: use_bias = True 

    tol = 0.01 if args.lp_early_stopping else 1e-4
    max_iter = 1000 if args.lp_early_stopping else 5000

    logger.info(f"Fitting Logistic Regression (max_iter={max_iter}, tol={tol})...")
    
    reg = LogisticRegressionCV(
        max_iter=max_iter, 
        fit_intercept=use_bias, 
        multi_class="multinomial", 
        random_state=args.seed, 
        tol=tol, 
        n_jobs=-1
    ).fit(features, targets)
    
    # 6. Assign learned weights back to LM Head
    logger.info("Fitting done. Assigning weights back to model...")
    decoder = _get_token_prediction_layer(model)
    
    coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
    if use_bias:
        bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)

    # Special handling: Binary classification returns a single vector, needs splitting
    if coef_torch.shape[0] == 1: 
        assert len(reg.classes_) == 2
        coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
        if use_bias:
            bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

    # Update only the relevant Token IDs
    with torch.no_grad():
        for _i, token_id in enumerate(reg.classes_):
            decoder.weight.data[token_id] = coef_torch[_i]
            if use_bias:
                decoder.bias.data[token_id] = bias_torch[_i]

    logger.info("Linear Probing finished.")