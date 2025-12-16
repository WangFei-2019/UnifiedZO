import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm

def perform_linear_probing(args, model, tokenizer, train_dataset, collator):
    """
    Executes Linear Probing logic strictly following the original ZO implementation.
    It extracts features, trains a logistic regression, and updates the LM head.
    """
    logger.info("Starting Linear Probing...")

    # 1. Helper to find the LM Head
    def _get_token_prediction_layer(model):
        # Support OPT and LLaMA as per original logic structure
        if "opt" in args.model_name_or_path.lower():
            return model.lm_head
        elif "llama" in args.model_name_or_path.lower() or "mistral" in args.model_name_or_path.lower():
            return model.lm_head
        else:
            # Fallback for generic causal LM, though original code specific to OPT
            return model.lm_head 

    # 2. Hook logic to extract features (pre-softmax/pre-head activations)
    # We need a container to store the captured feature from the forward hook
    captured_feats = {} 

    def __hook(model_, input_, output_):
        # input_[0] is the hidden state tensor fed into the LM head
        captured_feats["features"] = input_[0].detach()

    # Register the hook
    prediction_layer = _get_token_prediction_layer(model)
    handle = prediction_layer.register_forward_hook(__hook)

    # 3. Data Loader setup
    # We need a dataloader to iterate through the dataset with the correct collator
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        collate_fn=collator,
        shuffle=False # Sequential extraction
    )

    targets = []
    features = []

    logger.info("Extracting features from training set...")
    model.eval()
    
    with torch.inference_mode():
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Linear Probing Feature Extraction")):
            # Move inputs to device
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Forward pass triggers the hook
            # Note: We don't need the loss, just the forward pass to populate captured_feats
            outputs = model(**inputs)
            
            # Retrieve features from hook
            feature = captured_feats["features"] # shape: (batch, seq_len, hidden_dim)
            target = inputs["labels"]            # shape: (batch, seq_len)

            # --- Logic from Original ZO Code ---
            # Shift the target (bc it's autoregressive LM) and add the corresponding part
            # feature: t, target: t+1
            feature = feature[:, :-1, :]
            target = target[:, 1:]
            
            # Only extract the tokens corresponding to the option/answer
            # This relies on 'option_len' being present in the batch
            if "option_len" in inputs:
                for _i, _len in enumerate(inputs["option_len"]):
                    # Extract the last `_len` tokens
                    features.append(feature[_i, -_len:, :].cpu())
                    targets.append(target[_i, -_len:].cpu())
            else:
                # Fallback if option_len is missing (though ZO usually requires it)
                features.append(feature.reshape(-1, feature.shape[-1]).cpu())
                targets.append(target.reshape(-1).cpu())

    # Remove the hook after extraction
    handle.remove()

    # 4. Prepare data for Scikit-Learn
    features = torch.cat(features, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    
    logger.info(f"Feature shape: {features.shape}, Target shape: {targets.shape}")

    # 5. Configuration for Logistic Regression
    # Check if we should use bias (Original code: OPT/GPT2 -> False)
    use_bias = False 
    if "bert" in args.model_name_or_path: 
        use_bias = True # Just in case, though usually False for ZO LMs

    # Early stopping parameters from original code
    tol = 0.01 if args.lp_early_stopping else 1e-4
    max_iter = 1000 if args.lp_early_stopping else 5000

    logger.info(f"Fitting Logistic Regression (max_iter={max_iter}, tol={tol})...")
    
    # 6. Fit Model
    reg = LogisticRegressionCV(
        max_iter=max_iter, 
        fit_intercept=use_bias, 
        multi_class="multinomial", 
        random_state=args.seed, 
        tol=tol, 
        n_jobs=-1
    ).fit(features, targets)
    
    logger.info("Fitting done. Assigning weights back to model...")

    # 7. Assign weights back to LM Head
    decoder = _get_token_prediction_layer(model)
    
    # Convert sklearn weights to torch tensors on the correct device
    coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
    
    if use_bias:
        bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)

    # Special handling for binary classification in sklearn vs LM vocab
    # If regressor only detects two classes, coef_ shape might be (1, hidden)
    if coef_torch.shape[0] == 1: 
        assert len(reg.classes_) == 2
        # Original logic: split single vector into two opposite vectors
        coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
        if use_bias:
            bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

    # Map the learned weights to the specific token IDs in the vocabulary
    with torch.no_grad():
        for _i, token_id in enumerate(reg.classes_):
            decoder.weight.data[token_id] = coef_torch[_i]
            if use_bias:
                decoder.bias.data[token_id] = bias_torch[_i]

    logger.info("Linear Probing finished.")
