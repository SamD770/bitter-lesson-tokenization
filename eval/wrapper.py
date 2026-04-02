"""
lm_eval wrapper for byte-level language models.

This module provides a wrapper class that allows byte-level models (like AutoregressiveUnet)
to be evaluated using the EleutherAI lm-evaluation-harness.
"""

import torch
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


class ByteLevelLMWrapper(LM):
    """
    Wrapper for byte-level language models to work with lm_eval.
    
    This wrapper implements the LM interface required by lm_eval, translating
    requests into the format expected by the AutoregressiveUnet model.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        batch_size: int = 1,
        max_length: int = 4096,
        device: Optional[str] = None,
    ):
        """
        Initialize the wrapper.
        
        Args:
            model: A byte-level language model (e.g., AutoregressiveUnet)
            tokenizer: A byte-level tokenizer (e.g., EvaByte tokenizer)
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            device: Device to run inference on (defaults to model's device)
        """
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        
        if device is None:
            # Try to infer device from model parameters
            try:
                self._device = next(model.parameters()).device
            except StopIteration:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
    
    @property
    def eot_token_id(self) -> int:
        """End of text token ID."""
        if hasattr(self._tokenizer, 'eos_token_id') and self._tokenizer.eos_token_id is not None:
            return self._tokenizer.eos_token_id
        # Fallback for byte-level tokenizers that may not have explicit EOS
        return self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
    
    @property
    def max_length(self) -> int:
        """Maximum sequence length the model can handle."""
        return self._max_length
    
    @property
    def max_gen_toks(self) -> int:
        """Maximum tokens to generate."""
        return 256
    
    @property
    def batch_size(self) -> int:
        """Batch size for evaluation."""
        return self._batch_size
    
    @property
    def device(self) -> torch.device:
        """Device the model is on."""
        return self._device
    
    def tok_encode(self, string: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a string into token IDs."""
        return self._tokenizer.encode(
            string, 
            add_special_tokens=add_special_tokens,
            return_tensors=None
        )
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token IDs into a string."""
        return self._tokenizer.decode(tokens)
    
    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the model and return log-probabilities.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len]

        Returns:
            Log-probabilities of shape [batch, seq_len, vocab_size]
        """
        with torch.no_grad():
            from bpe_tokenizer.bpe_tokenizer import BPEAutoRegressiveUnet
            if isinstance(self._model, BPEAutoRegressiveUnet):
                texts = self._tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                out = self._model(input_ids, texts=texts)
            else:
                out = self._model(input_ids)
            # The model returns log-softmax logits
            return out["logits"]
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for (context, continuation) pairs.
        
        This is the key method for multiple-choice tasks like PIQA.
        
        Args:
            requests: List of Instance objects with (context, continuation) pairs
            
        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []
        
        # Process requests in batches
        for i in tqdm(range(0, len(requests), self._batch_size), desc="loglikelihood"):
            batch_requests = requests[i:i + self._batch_size]
            batch_results = self._compute_loglikelihood_batch(batch_requests)
            results.extend(batch_results)
        
        return results
    
    def _compute_loglikelihood_batch(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for a batch of requests.
        """
        results = []
        
        for request in requests:
            context, continuation = request.args
            
            # Encode context and continuation separately
            # We don't add special tokens to continuation to avoid double BOS
            context_ids = self.tok_encode(context, add_special_tokens=True)
            continuation_ids = self.tok_encode(continuation, add_special_tokens=False)
            
            # Combine them
            full_ids = context_ids + continuation_ids
            
            # Truncate if necessary (keep the end, which has the continuation)
            if len(full_ids) > self._max_length:
                # Calculate how much to truncate from context
                overflow = len(full_ids) - self._max_length
                context_ids = context_ids[overflow:]
                full_ids = context_ids + continuation_ids
            
            context_len = len(context_ids)
            
            # Convert to tensor
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self._device)
            
            # Get model output (log-probabilities)
            logits = self._model_call(input_tensor)  # [1, seq_len, vocab]
            
            # Compute log-likelihood for continuation tokens
            # For position i, logits[i] predicts token at position i+1
            # So for continuation starting at context_len, we look at logits[context_len-1:]
            log_likelihood = 0.0
            is_greedy = True
            
            for j, cont_token in enumerate(continuation_ids):
                # Position in the full sequence where we predict this token
                pos = context_len - 1 + j
                
                if pos >= logits.shape[1]:
                    break
                
                # Get log-prob for the actual continuation token
                token_logprob = logits[0, pos, cont_token].item()
                log_likelihood += token_logprob
                
                # Check if this was the greedy choice
                if logits[0, pos].argmax().item() != cont_token:
                    is_greedy = False
            
            results.append((log_likelihood, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute rolling log-likelihood (unconditional likelihood of text).
        
        This is used for perplexity computation.
        """
        results = []
        
        for request in tqdm(requests, desc="loglikelihood_rolling"):
            (text,) = request.args
            
            # Encode the full text
            token_ids = self.tok_encode(text, add_special_tokens=True)
            
            # Truncate if necessary
            if len(token_ids) > self._max_length:
                token_ids = token_ids[:self._max_length]
            
            # Convert to tensor
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self._device)
            
            # Get model output
            logits = self._model_call(input_tensor)
            
            # Compute total log-likelihood
            log_likelihood = 0.0
            for i in range(len(token_ids) - 1):
                next_token = token_ids[i + 1]
                log_likelihood += logits[0, i, next_token].item()
            
            # is_greedy is not meaningful for rolling, return True as placeholder
            results.append((log_likelihood, True))
        
        return results
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until a stop condition is met.
        
        This is used for free-form generation tasks.
        """
        raise NotImplementedError("generate_until is not implemented")
        results = []
        
        for request in tqdm(requests, desc="generate_until"):
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            
            # Get generation parameters
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            
            # Encode context
            context_ids = self.tok_encode(context, add_special_tokens=True)
            
            # Generate tokens autoregressively
            generated_ids = self._generate(
                context_ids,
                max_new_tokens=max_gen_toks,
                stop_sequences=until
            )
            
            # Decode only the generated part
            generated_text = self.tok_decode(generated_ids[len(context_ids):])
            
            # Truncate at stop sequences
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
            
            results.append(generated_text)
        
        return results
    
    def _generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 256,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> List[int]:
        """
        Generate tokens autoregressively.
        """
        raise NotImplementedError("generate is not implemented")
        if stop_sequences is None:
            stop_sequences = []
        
        current_ids = list(input_ids)
        
        for _ in range(max_new_tokens):
            # Truncate if we exceed max length
            if len(current_ids) >= self._max_length:
                break
            
            # Get model prediction for next token
            input_tensor = torch.tensor([current_ids], dtype=torch.long, device=self._device)
            logits = self._model_call(input_tensor)
            
            # Get next token (greedy for now, could add temperature sampling)
            next_token_logits = logits[0, -1, :]
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_token_logits.argmax().item()
            
            current_ids.append(next_token)
            
            # Check for EOS
            if next_token == self.eot_token_id:
                break
            
            # Check for stop sequences
            current_text = self.tok_decode(current_ids[len(input_ids):])
            if any(stop_seq in current_text for stop_seq in stop_sequences):
                break
        
        return current_ids
