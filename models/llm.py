"""Frozen LLM wrapper for text-based anomaly reasoning.

Accepts projected visual tokens and a text prompt, generates a
natural-language explanation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class FrozenLLM(nn.Module):
    """Wrapper around a frozen causal LM (e.g. Phi-2).

    Visual tokens are prepended to the text-token embeddings so the LLM
    can condition its generation on the visual information.
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

    # ── helpers ───────────────────────────────
    @property
    def embed_dim(self) -> int:
        """LLM hidden dimension."""
        return self.model.config.hidden_size

    def _get_text_embeddings(self, text: str, device: torch.device) -> torch.Tensor:
        """Tokenise a prompt and return its token embeddings.

        Returns:
            (1, T, D_llm) embedding tensor.
        """
        tokens = self.tokenizer(text, return_tensors="pt").input_ids.to(device)
        embed_layer = self.model.get_input_embeddings()
        return embed_layer(tokens)  # (1, T, D_llm)

    # ── forward / generate ────────────────────
    @torch.no_grad()
    def generate(
        self,
        visual_tokens: torch.Tensor,
        prompt: str,
    ) -> list[str]:
        """Generate text conditioned on visual tokens.

        Args:
            visual_tokens: (B, Q, D_llm) projected visual tokens.
            prompt: text prompt to append after visual tokens.

        Returns:
            List of B generated strings.
        """
        # Get device from model (LLM might be on GPU or CPU)
        model_device = next(self.model.parameters()).device
        B = visual_tokens.shape[0]

        # Move visual_tokens to model device
        visual_tokens = visual_tokens.to(model_device)

        # Get text embeddings for the prompt
        text_embeds = self._get_text_embeddings(prompt, model_device)  # (1, T, D)
        text_embeds = text_embeds.expand(B, -1, -1)                    # (B, T, D)

        # Concatenate: [visual_tokens | text_prompt_tokens]
        input_embeds = torch.cat(
            [visual_tokens.to(text_embeds.dtype), text_embeds], dim=1
        )  # (B, Q+T, D)

        # Create attention mask (all ones)
        attn_mask = torch.ones(
            input_embeds.shape[:2], dtype=torch.long, device=model_device,
        )

        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode generated tokens (skip the input portion)
        generated = []
        for i in range(B):
            text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            generated.append(text)

        return generated



