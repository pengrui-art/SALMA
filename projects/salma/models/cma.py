import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention2D(nn.Module):
    """
    Cross-modal attention block for injecting text semantics into 2D visual features.

    Inputs:
    - visual: (B, C, H, W)
    - text: (B, C) or (B, N, C)

    Features:
    - Multi-token support (no forced mean-pooling).
    - Top-k token routing (per batch) to keep compute cheap and focused.
    - Optional FiLM branch (text → per-channel affine) to stabilize and complement attention.
    - Gated residuals and warmup-friendly scaling hooks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_multi_token: bool = False,
        topk_tokens: int = 0,
        use_film: bool = False,
        film_dropout: float = 0.0,
        # Mask-biased options
        enable_mask_bias: bool = False,
        mask_bias_mode: str = "gate",  # 'gate' | 'bias' (bias not implemented yet)
        mask_bias_tau: float = 1.0,
        mask_bias_strength: float = 1.0,
        # Phrase-level routing options
        use_phrase_routing: bool = False,
        phrase_chunk_size: int = 0,  # 0 -> auto (2 or 3)
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_multi_token = use_multi_token
        self.topk_tokens = topk_tokens
        self.use_film = use_film
        self.enable_mask_bias = bool(enable_mask_bias)
        self.mask_bias_mode = str(mask_bias_mode)
        self.mask_bias_tau = float(mask_bias_tau)
        self.mask_bias_strength = float(mask_bias_strength)
        self.use_phrase_routing = bool(use_phrase_routing)
        self.phrase_chunk_size = int(phrase_chunk_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.ln = nn.LayerNorm(dim)
        # Output projection for attention path
        self.proj = nn.Linear(dim, dim)
        # Gating (initialized to 0) — attn path
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        # Backward-compat alias
        self.gamma = self.gamma_attn

        # Router MLP (shared for token or phrase scoring)
        if self.topk_tokens and self.topk_tokens > 0:
            self.token_gate = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.ReLU(inplace=True), nn.Linear(dim // 2, 1)
            )
        else:
            self.token_gate = None

        # Optional FiLM branch (text → per-channel affine)
        if self.use_film:
            self.film_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(film_dropout),
                nn.Linear(dim, dim * 2),
            )
            self.gamma_film = nn.Parameter(torch.zeros(1))
        else:
            self.film_mlp = None
            self.gamma_film = None

        # External warmup scale (multiplies gamma_* during forward)
        self.register_buffer("_gamma_scale", torch.ones(1))

    @torch.no_grad()
    def set_warmup_scale(self, scale: float):
        """Set an external scale for gating (e.g., during warmup). 1.0 means no scaling."""
        scale = float(scale)
        self._gamma_scale.fill_(scale)

    def _prepare_mask_spatial(
        self,
        mask_logits: torch.Tensor,
        H: int,
        W: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Validate and resize mask logits to (B, H, W) in the requested dtype."""
        m = mask_logits
        if m.dim() == 4 and m.size(1) == 1:
            m = m[:, 0]
        if m.dim() != 3:
            raise ValueError(
                f"mask_logits must be (B,H,W) or (B,1,H,W), got {tuple(mask_logits.shape)}"
            )
        if (m.shape[-2] != H) or (m.shape[-1] != W):
            m = F.interpolate(
                m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)
        return m.to(dtype=dtype)

    def _select_text_tokens(self, text: torch.Tensor) -> torch.Tensor:
        """Return text tokens as (B, Nt, C). Supports (B,C) or (B,N,C).
        - If use_multi_token=False, fallback to mean-pooled single token.
        - If phrase routing enabled, group tokens into phrases (simple chunks) and route top-k phrases.
        - Else if topk_tokens>0, select top-k tokens.
        """
        if text.dim() == 2:
            return text[:, None, :]  # (B,1,C)
        if text.dim() != 3:
            raise ValueError(f"Unsupported text shape: {text.shape}")

        if not self.use_multi_token:
            return text.mean(dim=1, keepdim=True)  # (B,1,C)

        # Phrase-level routing path (chunk grouping). If disabled, fallback to token routing.
        if self.use_phrase_routing and self.token_gate is not None:
            B, N, C = text.shape
            chunk = (
                self.phrase_chunk_size
                if self.phrase_chunk_size > 0
                else (3 if N >= 6 else 2)
            )
            # Build phrase embeddings by average pooling fixed-size chunks.
            num_chunks = (N + chunk - 1) // chunk
            # Pad tokens to multiple of chunk size if needed.
            if N % chunk != 0:
                pad_needed = chunk - (N % chunk)
                pad = text[:, -1:].expand(B, pad_needed, C)
                text_padded = torch.cat([text, pad], dim=1)
            else:
                text_padded = text
            phrases = text_padded.view(B, -1, chunk, C).mean(
                dim=2
            )  # (B, num_chunks, C)
            gate_w_dtype = next(self.token_gate.parameters()).dtype
            scores = self.token_gate(phrases.to(dtype=gate_w_dtype)).squeeze(
                -1
            )  # (B,num_chunks)
            k = (
                min(self.topk_tokens, phrases.size(1))
                if self.topk_tokens > 0
                else phrases.size(1)
            )
            topk_idx = scores.topk(
                k=k, dim=1, largest=True, sorted=False
            ).indices  # (B,k)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, C)
            picked_phrases = torch.gather(phrases, dim=1, index=idx_exp)  # (B,k,C)
            return picked_phrases

        # Token-level routing (original behavior)
        if self.token_gate is None:
            return text  # (B,N,C)
        B, N, C = text.shape
        gate_w_dtype = next(self.token_gate.parameters()).dtype
        scores = self.token_gate(text.to(dtype=gate_w_dtype)).squeeze(-1)  # (B,N)
        k = min(self.topk_tokens, N) if self.topk_tokens > 0 else N
        topk_idx = scores.topk(k=k, dim=1, largest=True, sorted=False).indices  # (B,k)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        picked = torch.gather(text, dim=1, index=idx_exp)
        return picked

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        mask_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = visual.shape

        # Prepare Q from visual tokens
        x = visual.flatten(2).transpose(1, 2)  # (B, HW, C)
        x_ln = F.layer_norm(
            x.float(),
            normalized_shape=(C,),
            weight=self.ln.weight.float() if self.ln.weight is not None else None,
            bias=self.ln.bias.float() if self.ln.bias is not None else None,
            eps=self.ln.eps,
        ).to(dtype=x.dtype)
        q = x_ln.transpose(0, 1)  # (HW, B, C)

        # Prepare K/V from text (supports multi-token + routing)
        t = self._select_text_tokens(text)  # (B,Nt,C)
        k = t.transpose(0, 1)  # (Nt, B, C)
        v = k

        attn_weight_dtype = next(self.attn.parameters()).dtype
        # Prepare optional mask bias map for attention / gating
        mask_sigmoid = None
        attn_bias = None
        if self.enable_mask_bias and mask_logits is not None:
            try:
                mask_spatial = self._prepare_mask_spatial(
                    mask_logits, H, W, dtype=attn_weight_dtype
                )
                tau = max(self.mask_bias_tau, 1e-6)
                mask_sigmoid = torch.sigmoid(mask_spatial / tau)
                if self.mask_bias_mode == "bias":
                    eps = 1e-4
                    bias = torch.logit(mask_sigmoid.clamp(min=eps, max=1.0 - eps))
                    bias = bias * float(self.mask_bias_strength)
                    bias = bias.flatten(1).unsqueeze(-1).expand(-1, -1, t.size(1))
                    bias = bias.unsqueeze(1).repeat_interleave(self.num_heads, dim=1)
                    attn_bias = bias.reshape(B * self.num_heads, H * W, t.size(1)).to(
                        dtype=attn_weight_dtype
                    )
            except Exception:
                mask_sigmoid = None
                attn_bias = None

        # Cross attention
        q_cast = q.to(dtype=attn_weight_dtype)
        k_cast = k.to(dtype=attn_weight_dtype)
        v_cast = v.to(dtype=attn_weight_dtype)
        # NOTE: additive attention bias path is reserved for future
        # For now, compute attention normally and optionally apply spatial gating using mask logits
        if attn_bias is not None:
            attn_out, _ = self.attn(q_cast, k_cast, v_cast, attn_mask=attn_bias)
        else:
            attn_out, _ = self.attn(q_cast, k_cast, v_cast)
        attn_out = attn_out.to(dtype=q.dtype).transpose(0, 1)  # (B, HW, C)

        # Linear proj for attention path
        proj_w_dtype = self.proj.weight.dtype
        attn_out = self.proj(attn_out.to(dtype=proj_w_dtype)).to(dtype=q.dtype)

        # Optional mask-biased spatial gating (gate mode)
        if mask_sigmoid is not None and self.mask_bias_mode == "gate":
            gate = mask_sigmoid.to(dtype=attn_out.dtype)
            strength = max(float(self.mask_bias_strength), 0.0)
            if strength not in (0.0, 1.0):
                gate = gate.pow(strength)
            elif strength == 0.0:
                gate = torch.ones_like(gate)
            gate = gate.flatten(1).unsqueeze(-1)
            attn_out = attn_out * gate

        # Residual with gating (attention)
        gamma_attn = (self.gamma_attn * self._gamma_scale).to(dtype=x.dtype)
        x = x + gamma_attn * attn_out.to(dtype=x.dtype)

        # Optional FiLM branch
        if self.use_film and self.film_mlp is not None:
            # Recompute norm after attention residual
            x_ln2 = F.layer_norm(
                x.float(),
                normalized_shape=(C,),
                weight=self.ln.weight.float() if self.ln.weight is not None else None,
                bias=self.ln.bias.float() if self.ln.bias is not None else None,
                eps=self.ln.eps,
            ).to(dtype=x.dtype)
            # Global text summary for FiLM (mean over tokens)
            if text.dim() == 2:
                t_global = text  # (B,C)
            else:
                t_global = text.mean(dim=1)  # (B,C)
            # Dtype-safe FiLM MLP
            film_w_dtype = next(self.film_mlp.parameters()).dtype
            film_params = self.film_mlp(t_global.to(dtype=film_w_dtype))  # (B,2C)
            gamma_c, beta_c = film_params.split(C, dim=-1)  # (B,C),(B,C)
            gamma_c = gamma_c.to(dtype=x.dtype)
            beta_c = beta_c.to(dtype=x.dtype)
            # Token-wise apply FiLM: (B,HW,C)
            film = x_ln2 * gamma_c[:, None, :] + beta_c[:, None, :]
            gamma_film = (self.gamma_film * self._gamma_scale).to(dtype=x.dtype)
            x = x + gamma_film * film

        # Restore spatial shape
        out = x.transpose(1, 2).reshape(B, C, H, W)
        return out
