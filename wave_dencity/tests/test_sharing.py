import torch
from train_transplant import (
    _load_causal_lm,
    patch_attention_modules,
    ParallelAttentionWrapper,
    WDABridge,
)


class FakeCfg:
    def __init__(self):
        self.seq_len = 128
        self.wda_num_masks = 8
        self.wda_num_waves_per_mask = 4
        self.wda_topk_masks = 4
        self.wda_attn_alpha = 3.0
        self.wda_content_mix = 0.1
        self.wda_learned_content = True
        self.wda_use_sin_waves = True
        self.wda_use_sampling = False
        self.wda_num_samples = 32
        self.wda_noise_sigma = 0.12
        self.wda_step_alpha = 6.0
        self.wda_use_checkpoint = False
        self.alpha_layer_scale = "uniform"
        self.init_gamma = 0.44


def test_sharing():
    cfg = FakeCfg()

    # Mocking a small model or just parts of it
    class MockLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(10, 10)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([MockLayer(), MockLayer()])

    model = MockModel()

    def make_wrapper(attn_mod, path):
        wda = WDABridge(
            hidden_size=10,
            num_heads=2,
            seq_len=cfg.seq_len,
            num_masks=cfg.wda_num_masks,
            num_waves_per_mask=cfg.wda_num_waves_per_mask,
            topk_masks=cfg.wda_topk_masks,
            attn_alpha=cfg.wda_attn_alpha,
        )
        return ParallelAttentionWrapper(
            attn_mod,
            wda,
            layer_idx=0,  # dummy
            init_gamma=cfg.init_gamma,
        )

    patched = patch_attention_modules(model, make_wrapper)
    print(f"Patched: {patched}")

    gammas = []
    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            gammas.append(m.gamma)

    if len(gammas) > 1:
        is_same = gammas[0] is gammas[1]
        print(f"Gamma 0: {id(gammas[0])}")
        print(f"Gamma 1: {id(gammas[1])}")
        print(f"Shared? {is_same}")
    else:
        print("Not enough gammas found")


if __name__ == "__main__":
    test_sharing()
