import torch
from torch import nn
import torch.nn.functional as F
import math
from optimizing_code.triton_scan_simple import torch_scan_simple, compute_probs, torch_scan_simple_scaled, compute_probs_scaled


class SequentialyDependentGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float, filter_size: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.filter_size = filter_size
        raise NotImplementedError("Use ScaledSequentialyDependentLinearGater instead.")

    def forward(self, x: torch.Tensor, downsample_rate: float) -> torch.Tensor:
        """
        Computes the conditional probablilities and samples from them.
        WARNING: For now, the downsample rate is ignored.
        Returns:
        gate_logits[i]: log_p(gate_samples[i] | gate_samples[:i])
        gate_probs[i]: p(gate_samples[i] | gate_samples[:i])
        gate_samples[i]
        It's 22:30 and I can't find the sneaky way to do this with cumsum. Could also be done with jax.lax.scan or CUDA if it works.
        """
        batch_size, seq_len, _ = x.shape

        filter_weights = self.precompute_filter_weights(x)

        logits_list = []
        probs_list = []
        samples_list = [torch.zeros(batch_size, 1, 1, dtype=x.dtype, device=x.device) for _ in range(self.filter_size)]

        
        for seq_idx in range(seq_len):

            weight = self.scan_filter_weights(samples_list, filter_weights, batch_size, seq_idx)

            log_probs = F.logsigmoid(weight)
            probs = F.sigmoid(weight)
            samples = torch.bernoulli(probs)

            logits_list.append(log_probs)
            probs_list.append(probs)
            samples_list.append(samples)
        
        gate_logits = torch.cat(logits_list, dim=1)
        gate_probs = torch.cat(probs_list, dim=1)
        # Remove the padding
        samples_list = samples_list[self.filter_size:]
        gate_samples = torch.cat(samples_list, dim=1)

        return gate_logits, gate_probs, gate_samples

    def precompute_filter_weights(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def scan_filter_weights(self, samples_list: torch.Tensor, filter_weights: torch.Tensor, batch_size: int, seq_idx: int) -> torch.Tensor:
        raise NotImplementedError()

        
# This gater is used to conditionally gate the tokens.
class SequentiallyDependentRandomGater(SequentialyDependentGater):
    def __init__(self, embedding_dim: int, downsample_rate: float, filter_size: int = 4):
        super().__init__(embedding_dim=embedding_dim, downsample_rate=downsample_rate, filter_size=filter_size)
        base_logit_init = torch.log(torch.tensor(downsample_rate / (1 - downsample_rate))) # Use the inverse sigmoid to get the logit of the downsample rate.
        self.base_value = nn.Parameter(torch.ones((1,)) * base_logit_init)
        self.filter = nn.ParameterList([nn.Parameter(torch.zeros((1,))) for _ in range(filter_size)])
        raise NotImplementedError("Use ScaledSequentialyDependentLinearGater instead.")

    def precompute_filter_weights(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def scan_filter_weights(self, samples_list: torch.Tensor, filter_weights: torch.Tensor, batch_size: int, seq_idx: int) -> torch.Tensor:

        weight = self.base_value * torch.ones(batch_size, 1, 1, dtype=self.base_value.dtype, device=self.base_value.device)
        
        for i in range(self.filter_size):
            weight += self.filter[i] * samples_list[-i-1]
        
        return weight


class SequentiallyDependentLinearGater(SequentialyDependentGater):
    def __init__(self, embedding_dim: int, downsample_rate: float, filter_size: int = 8):
        super().__init__(embedding_dim=embedding_dim, downsample_rate=downsample_rate, filter_size=filter_size)
        self.filter_layer = nn.Linear(embedding_dim, filter_size + 1) # +1 for the base value
        raise NotImplementedError("Use ScaledSequentialyDependentLinearGater instead.")

    def precompute_filter_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the filter weights for each token in the sequence. Of shape (batch_size, seq_len, filter_size + 1)"""
        return self.filter_layer(x)
    
    def scan_filter_weights(self, samples_list: torch.Tensor, filter_weights: torch.Tensor, batch_size: int, seq_idx: int) -> torch.Tensor:
        base_values = filter_weights[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        filter_values = filter_weights[:, :, 1:].unsqueeze(-1).unsqueeze(-1)

        weight = base_values[:, seq_idx] * torch.ones(batch_size, 1, 1, dtype=base_values.dtype, device=base_values.device)
        
        for i in range(self.filter_size):
            weight += filter_values[:, seq_idx, i] * samples_list[-i-1]
        
        return weight


class OptimizedSequentialyDependentLinearGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float, filter_size: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate
        self.filter_size = filter_size
        self.filter_layer = nn.Linear(embedding_dim, filter_size)
        raise NotImplementedError("Use ScaledSequentialyDependentLinearGater instead.")

    def forward(self, x: torch.Tensor, downsample_rate: float) -> torch.Tensor:
        V = self.filter_layer(x)
        scan_logits, scan_probs, a = torch_scan_simple(V)
        logits, probs = compute_probs(a, V)
        return logits.unsqueeze(-1), probs.unsqueeze(-1), a.unsqueeze(-1)


class ScaledSequentialyDependentLinearGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float, filter_size: int = 8, scale_factor: float = 1/16.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate
        self.filter_size = filter_size
        self.scale_factor = scale_factor
        self.filter_layer = nn.Linear(embedding_dim, filter_size)

    def forward(self, x: torch.Tensor, downsample_rate: float) -> torch.Tensor:
        
        if downsample_rate is None:
            downsample_rate = self.downsample_rate

        downsample_rate = min(downsample_rate, 0.999) # In case the downsample rate is 1.
        bias = math.log(downsample_rate / (1 - downsample_rate))
        x_init_dtype = x.dtype

        V = self.filter_layer(x)
        # During training, we use the soft-cap, during evaluation, we don't.
        scan_logits, scan_probs, a = torch_scan_simple_scaled(V, scale_factor=self.scale_factor, bias=bias, use_soft_cap=self.training)
        logits, probs = compute_probs_scaled(a, V, scale_factor=self.scale_factor, bias=bias, use_soft_cap=self.training)
        
        logits = logits.to(dtype=x_init_dtype)
        probs = probs.to(dtype=x_init_dtype)
        a = a.to(dtype=x_init_dtype)

        
        return logits.unsqueeze(-1), probs.unsqueeze(-1), a.unsqueeze(-1)



if __name__ == "__main__":
    my_x = torch.randn(32, 1024, 256, dtype=torch.float32, device="cuda")
    gater = ScaledSequentiallyDependentLinearGater(embedding_dim=256, downsample_rate=0.25, filter_size=4).to("cuda", dtype=torch.float32)

    print(gater)

    import time
    my_optimizer = torch.optim.Adam(gater.parameters(), lr=0.1)

    total_time = 0
    n_iters = 20
    for i in range(n_iters + 1):
        start_time = time.time()
        gate_logits, gate_probs, gate_samples = gater(my_x)
        loss = 10**4. * ((gate_probs).mean() -  0.25)**2
        loss.backward()
        my_optimizer.step()
        my_optimizer.zero_grad()
        end_time = time.time()
        
        # Skip the iteration with compilation
        if i != 0:
            total_time += (end_time - start_time)

   
    print(f"{gate_samples.shape=} {gate_probs.shape=} {gate_logits.shape=}")

    print(f"{gate_probs.mean().item()=}")
    print(f"{gate_samples[:4, :4, 0]=}")
    print(f"{gate_probs[:4, :4, 0]=}")
    print(f"{gate_logits[:4, :4, 0]=}")


    # for i in range(gater.filter_size):
    #     print(f"{gater.filter[i].data=}")

    # print(f"{gater.base_value.data=}")
    
    print(f"Average time per forward/backward pass: {total_time / n_iters:.4f} seconds")
    print(f"Total time for {n_iters} passes: {total_time:.4f} seconds")
 