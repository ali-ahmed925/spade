"""SPADE — anomaly detection over contextualised multi-scale patch descriptors.

FLOW
----
    image (B, 3, 448, 448)
      -> frozen ViT-G, position embeddings interpolated 224 -> 448
           taps blocks 20 and 30 in addition to the final block
      -> (B, 1024, 1408) per tapped block                       [models/vit.py]
      -> MultiLayerPatchFusion: per-block LayerNorm + projection + concat
      -> (B, 1024, 512) fused patch features           [models/feature_fusion.py]
      -> Q-Former over the final block  ->  (B, 32, 768) query tokens
      -> QueryPatchContextualizer: every patch attends to the 32 queries
      -> (B, 1024, 512) CONTEXTUALISED descriptors     [models/feature_fusion.py]
      -> Mahalanobis over descriptors           [models/mahalanobis_scoring.py]
       + parallel Fourier frequency stream      [models/frequency_features.py]
      -> (B, 1024) patch scores -> 32x32 map -> heatmap        [utils/heatmap.py]
      -> LLMProjection(query tokens) -> visual tokens for language [models/llm.py]

WHY THIS SPACE
--------------
Mahalanobis previously scored the frozen FINAL-layer ViT embedding of each patch,
independently of every other patch, on a 16x16 grid. Measured consequences:

  * one patch covered 196 px, so capsule crack (0.30% of the image), poke (0.23%)
    and faulty_imprint (0.28%) were each SMALLER THAN ONE PATCH and were averaged
    away before scoring — 94% of capsule's AUROC deficit;
  * the final block is BLIP-2's most semantic representation, invariant to
    exactly the sub-semantic damage we need (defect elevation was 1.5-3x on weak
    classes against 8-75x on classes that work);
  * patch independence makes role violations undetectable in principle —
    cable_swap is ordinary blue insulation in the wrong position, cut_lead is
    ordinary board where a lead should be. Both are individually normal, and
    cut_lead's defect patches actually scored BELOW that image's clean patches
    (within-image AUROC 0.436).

Resolution and layer choice address the first two; Q-Former contextualisation
addresses the third, in one descriptor, with no per-defect-type logic.

Trainable: fusion, contextualizer, Q-Former, projection.
Frozen: ViT-G, LLM.
Fitted in closed form: Mahalanobis mu/Sigma over descriptors (not learned by
gradient — accumulated by NormalStatisticsTracker and refitted periodically).
"""

import torch
import torch.nn as nn
from transformers import Blip2Model

from models.memory_bank import CoresetMemoryBank
from models.neighborhood import NeighborhoodAggregator
from models.feature_fusion import MultiLayerPatchFusion, QueryPatchContextualizer
from models.mahalanobis_scoring import MahalanobisScoring
from models.normal_statistics import NormalStatisticsTracker
from models.projection import LLMProjection
from models.qformer import Blip2QFormerWrapper
from models.vit import FrozenVisionEncoder
from utils.debug_logger import get_debug_logger

_debug_logger = get_debug_logger("spade_debug")


class SPADE(nn.Module):
    """Full SPADE model over contextualised multi-scale patch descriptors."""

    def __init__(
        self,
        blip2_model_name: str = "Salesforce/blip2-opt-2.7b",
        llm_embed_dim: int = 2560,
        # ── vision front-end ──
        image_size: int = 448,
        feature_layers: tuple[int, ...] | list[int] = (20, 30),
        fusion_proj_dim: int = 256,
        # ── Q-Former contextualisation ──
        context_hidden_dim: int = 256,
        context_heads: int = 8,
        context_dropout: float = 0.0,
        # ── scoring ──
        score_beta: float = 0.9,
        score_gamma: float = 0.1,
        mahalanobis_gamma: float = 1.0,
        mahalanobis_reg: float = 1e-4,
        normalize_streams: bool = True,
        image_aggregation: str = "topk_mean",
        # ── local detection pathway ──
        local_enabled: bool = True,
        local_source: str = "fused",
        local_neighborhood: int = 3,
        score_w_local: float = 1.0,
        memory_bank_cfg: dict | None = None,
        # ── normal statistics ──
        normal_stats_buffer_size: int = 20000,
        normal_stats_update_frequency: int = 100,
        # ── misc ──
        projection_trainable: bool = False,
        blip2_model=None,
    ):
        super().__init__()

        # Load pretrained BLIP-2 (unless one was injected for testing)
        if blip2_model is not None:
            blip2 = blip2_model
        else:
            import os

            token = os.environ.get("HF_TOKEN")
            blip2 = Blip2Model.from_pretrained(
                blip2_model_name, token=token, torch_dtype=torch.float16
            )

        # ── frozen vision encoder at the configured resolution ──
        self.vision_encoder = FrozenVisionEncoder(
            blip2.vision_model, image_size=image_size, feature_layers=feature_layers
        )
        # ── trainable Q-Former ──
        self.qformer = Blip2QFormerWrapper(blip2.qformer, blip2.query_tokens)

        del blip2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vit_dim = self.vision_encoder.hidden_size          # 1408
        qformer_dim = self.qformer.hidden_size             # 768
        self.image_size = self.vision_encoder.image_size   # 448
        self.grid_size = self.vision_encoder.grid_size     # 32
        self.num_patches = self.vision_encoder.num_patches # 1024

        # ── descriptor construction ──
        self.fusion = MultiLayerPatchFusion(
            in_dim=vit_dim,
            n_layers=len(self.vision_encoder.feature_layers),
            proj_dim=fusion_proj_dim,
        )
        self.contextualizer = QueryPatchContextualizer(
            patch_dim=self.fusion.output_dim,
            query_dim=qformer_dim,
            hidden_dim=context_hidden_dim,
            n_heads=context_heads,
            dropout=context_dropout,
        )
        self.descriptor_dim = self.contextualizer.output_dim

        # ── LOCAL detection pathway ──
        # The contextualizer OVERWRITES rather than augments: the only thing
        # Mahalanobis ever sees is [local | context], where the context half is
        # attention over 32 image-level tokens and is near-constant across
        # patches within an image. Under one pooled Gaussian that half varies
        # between images but not between patches, so it inflates Sigma without
        # adding discrimination -- and no raw local appearance reaches the score
        # without passing through global mixing first.
        #
        # This pathway runs in parallel and never touches the contextualizer.
        if local_source not in ("fused", "raw"):
            raise ValueError(f"local_source must be 'fused' or 'raw', got {local_source!r}")
        self.local_enabled = bool(local_enabled)
        self.local_source = local_source
        self.local_dim = (
            self.fusion.output_dim if local_source == "fused"
            else vit_dim * len(self.vision_encoder.feature_layers)
        )
        self.neighborhood = NeighborhoodAggregator(
            grid_size=self.grid_size, kernel_size=local_neighborhood
        )
        self.memory_bank = CoresetMemoryBank(
            feature_dim=self.local_dim, **(memory_bank_cfg or {})
        )

        # ── scoring over descriptors ──
        self.mahalanobis_scorer = MahalanobisScoring(
            feature_dim=self.descriptor_dim,
            regularization=mahalanobis_reg,
            gamma=mahalanobis_gamma,
        )
        self.normal_stats_tracker = NormalStatisticsTracker(
            feature_dim=self.descriptor_dim,
            buffer_size=normal_stats_buffer_size,
            update_frequency=normal_stats_update_frequency,
        )

        # ── language path ──
        self.projection = LLMProjection(input_dim=qformer_dim, output_dim=llm_embed_dim)
        self.projection_trainable = projection_trainable
        for p_ in self.projection.parameters():
            p_.requires_grad = bool(projection_trainable)

        # ── score weights ──
        self.image_aggregation = image_aggregation
        self.score_beta = score_beta
        self.score_gamma = score_gamma
        self.score_w_local = score_w_local

        # ── stream scaling ──
        # One global constant per stream, EMA-estimated during training and held
        # fixed at inference, so beta/gamma mean what they claim. Deliberately
        # NOT per-image: that would destroy global calibration (EVALUATION_FIX.md).
        self.normalize_streams = normalize_streams
        self.register_buffer("mahal_scale", torch.ones(()))
        self.register_buffer("freq_scale", torch.ones(()))
        # Fitted alongside the bank, not EMA'd: the bank does not exist during
        # training, so there is nothing to average over.
        self.register_buffer("local_scale", torch.ones(()))
        self.register_buffer("stream_scales_initialized", torch.tensor(False))
        self.stream_scale_momentum = 0.05

        # ── optional frequency stream ──
        self.use_frequency = False
        self.freq_extractor = None
        self.freq_mahalanobis_scorer = None
        self.freq_normal_stats_tracker = None

    # ──────────────────────────────────────────────────────────────────────
    # Descriptor construction
    # ──────────────────────────────────────────────────────────────────────
    def build_descriptors(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """image -> multi-layer patches -> Q-Former queries -> contextual descriptors.

        Returns a dict with "descriptors" (B, N, C), "query_embeds" (B, Q, E) and,
        when requested, "patch_query_attention" (B, N, Q).
        """
        encoded = self.vision_encoder(images, return_layers=True)
        final_hidden = encoded["final"]                       # (B, 1+N, D)
        layer_patches = [h[:, 1:, :] for h in encoded["layers"]]

        # Fused local appearance across scales.
        fused = self.fusion(layer_patches)                    # (B, N, F)

        # Image-level summary. The Q-Former is fed the final block INCLUDING CLS,
        # which is the input distribution it was pretrained on.
        query_embeds = self.qformer(final_hidden.float())     # (B, Q, E)

        contextual = self.contextualizer(
            fused, query_embeds, return_attention=return_attention
        )
        if return_attention:
            descriptors, attention = contextual
        else:
            descriptors, attention = contextual, None

        out = {"descriptors": descriptors, "query_embeds": query_embeds}
        if attention is not None:
            out["patch_query_attention"] = attention

        # The local pathway branches HERE, before the contextualizer, so nothing
        # global is mixed in. Neighbourhood pooling is applied identically when
        # fitting the bank and when scoring against it -- if it were applied to
        # only one, queries and bank vectors would live in different spaces.
        if self.local_enabled:
            local = fused if self.local_source == "fused" else torch.cat(layer_patches, dim=-1)
            out["local_features"] = self.neighborhood(local)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Score composition
    # ──────────────────────────────────────────────────────────────────────
    def _update_stream_scales(self, mahal: torch.Tensor, freq: torch.Tensor) -> None:
        """EMA the per-stream magnitudes so beta/gamma/w_local mean what they say.

        The local stream is absent here on purpose: the memory bank does not
        exist during training, so its scale is fitted in one shot by
        `fit_normal_model` instead of averaged over batches.
        """
        m = self.stream_scale_momentum
        with torch.no_grad():
            d = mahal.detach().abs().mean().clamp_min(1e-8)
            f = freq.detach().abs().mean().clamp_min(1e-8)
            if not bool(self.stream_scales_initialized):
                self.mahal_scale.fill_(float(d))
                self.freq_scale.fill_(float(f))
                self.stream_scales_initialized.fill_(True)
            else:
                self.mahal_scale.mul_(1 - m).add_(m * d)
                self.freq_scale.mul_(1 - m).add_(m * f)

    def _compose_scores(
        self,
        spatial_mahal: torch.Tensor,
        freq_mahal: torch.Tensor,
        local_knn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combine the local, contextual and frequency streams.

        Each stream is divided by ONE GLOBAL constant, never a per-image one --
        per-image normalization destroys the calibration that image-level AUROC
        depends on (EVALUATION_FIX.md).
        """
        if self.normalize_streams:
            spatial = spatial_mahal / self.mahal_scale.clamp_min(1e-8)
            frequency = freq_mahal / self.freq_scale.clamp_min(1e-8)
        else:
            spatial, frequency = spatial_mahal, freq_mahal

        components = {
            "contextual_mahalanobis": self.score_beta * spatial,
            "frequency": self.score_gamma * frequency,
        }
        if local_knn is not None:
            local = (
                local_knn / self.local_scale.clamp_min(1e-8)
                if self.normalize_streams else local_knn
            )
            components["local_knn"] = self.score_w_local * local

        total = sum(components.values())
        return total, components

    def _score_perturbed(
        self, descriptors: torch.Tensor, freq_mahal: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        """Score a noise-perturbed copy of the descriptors.

        Used by the pseudo-anomaly objective. The perturbation is applied to the
        descriptors the score genuinely depends on, not to the scores themselves:
        perturbing scores cancels algebraically and yields exactly zero gradient
        (see losses/patch_loss_normal.py).
        """
        scale = descriptors.detach().std().clamp_min(1e-8)
        perturbed = descriptors + torch.randn_like(descriptors) * (epsilon * scale)
        scores, _ = self._compose_scores(self.mahalanobis_scorer(perturbed), freq_mahal)
        return scores

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        images: torch.Tensor,
        patch_labels: torch.Tensor | None = None,
        update_stats: bool = False,
        perturb_epsilon: float | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, image_size, image_size) normalized batch.
            patch_labels: (B, N) binary patch labels, for statistics accumulation.
            update_stats: accumulate normal descriptors and refit Mahalanobis.
            perturb_epsilon: if set, also score a perturbed copy of the descriptors.
            return_attention: also return patch-to-query attention (B, N, Q).

        Returns:
            dict with patch_scores (B, N), descriptors, query_embeds,
            visual_tokens, score_components, and optionally
            patch_scores_perturbed / patch_query_attention.
        """
        built = self.build_descriptors(images, return_attention=return_attention)
        descriptors = built["descriptors"]
        query_embeds = built["query_embeds"]

        _debug_logger.debug(
            f"[SPADE] training={self.training} descriptors={tuple(descriptors.shape)} "
            f"grid={self.grid_size}x{self.grid_size}"
        )

        # ── normal statistics over descriptors ──
        if update_stats and patch_labels is not None:
            self.normal_stats_tracker.add_normal_patches(descriptors, patch_labels)
            tracker = self.normal_stats_tracker
            if tracker.step_count % tracker.update_frequency == 0:
                if len(tracker.normal_patch_buffer) > 0:
                    normal_descriptors = torch.stack(
                        list(tracker.normal_patch_buffer)
                    ).to(descriptors.device)
                    self.mahalanobis_scorer.update_statistics(normal_descriptors)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            tracker.step_count += 1

        # ── frequency stream (parallel, on raw image patches) ──
        freq_mahal = torch.zeros(
            descriptors.shape[0], descriptors.shape[1],
            device=descriptors.device, dtype=descriptors.dtype,
        )
        if self.use_frequency and self.freq_extractor is not None:
            from utils.patch_extraction import extract_image_patches_from_tensor

            image_patches = extract_image_patches_from_tensor(
                images, patch_size=self.vision_encoder.patch_size
            )
            patches_tensor = torch.from_numpy(image_patches).to(images.device)
            freq_features = self.freq_extractor(patches_tensor)
            freq_features = freq_features.reshape(
                descriptors.shape[0], self.num_patches, -1
            )

            if update_stats and patch_labels is not None:
                self.freq_normal_stats_tracker.add_normal_patches(
                    freq_features, patch_labels
                )
                freq_tracker = self.freq_normal_stats_tracker
                if freq_tracker.step_count % freq_tracker.update_frequency == 0:
                    if len(freq_tracker.normal_patch_buffer) > 0:
                        normal_freq = torch.stack(
                            list(freq_tracker.normal_patch_buffer)
                        ).to(freq_features.device)
                        self.freq_mahalanobis_scorer.update_statistics(normal_freq)
                freq_tracker.step_count += 1

            freq_mahal = self.freq_mahalanobis_scorer(freq_features)

        # ── Mahalanobis over the contextual descriptors ──
        spatial_mahal = self.mahalanobis_scorer(descriptors)   # (B, N)

        if self.training and update_stats:
            self._update_stream_scales(spatial_mahal, freq_mahal)

        # ── local pathway: nearest normal patch, no distributional assumption ──
        local_features = built.get("local_features")
        local_knn = None
        if local_features is not None and self.memory_bank.fitted:
            local_knn = self.memory_bank(local_features)

        patch_scores, score_components = self._compose_scores(
            spatial_mahal, freq_mahal, local_knn
        )

        _debug_logger.debug(
            f"[SPADE] mahal mean={float(spatial_mahal.detach().mean()):.4f} "
            f"freq mean={float(freq_mahal.detach().mean()):.4f} "
            f"score mean={float(patch_scores.detach().mean()):.4f}"
        )

        # ── language path ──
        visual_tokens = self.projection(query_embeds)

        outputs = {
            "patch_scores": patch_scores,
            "descriptors": descriptors,
            "query_embeds": query_embeds,
            "visual_tokens": visual_tokens,
            "score_components": score_components,
        }
        if "patch_query_attention" in built:
            outputs["patch_query_attention"] = built["patch_query_attention"]
        if local_features is not None:
            outputs["local_features"] = local_features
        if perturb_epsilon is not None:
            outputs["patch_scores_perturbed"] = self._score_perturbed(
                descriptors, freq_mahal, perturb_epsilon
            )
        return outputs

    # ──────────────────────────────────────────────────────────────────────
    def top_k_for(self, n_patches: int) -> int:
        """How many patches the top-k mean averages, scaled with the patch grid.

        top-3 of 256 patches covers 1.17% of the image; the equivalent at 1024
        patches is 12. Keeping k=3 would silently make the image score 4x more
        selective and make pre/post-redesign numbers incomparable.

        Note the tension this creates, which is why both aggregations are
        reported: a capsule crack spans ~3 patches at 448, so a top-12 mean
        averages 3 defect patches with 9 normal ones.
        """
        return max(1, min(round(3 * n_patches / 256), n_patches))

    def get_image_score(
        self,
        patch_scores: torch.Tensor,
        aggregation: str | None = None,
    ) -> torch.Tensor:
        """Reduce (B, N) patch scores to (B,) image scores.

        aggregation:
            "topk_mean"  mean of the top-k patches, k scaled to the grid.
            "max"        the single highest patch. This is what PaDiM,
                         PatchCore and the WACV SingleNet reference all use;
                         the top-k mean is the non-standard choice here.

        The choice is a METHOD component, not an evaluation protocol, so it must
        be declared and ablated rather than picked for whichever scores higher.
        eval.py reports both on every run.
        """
        mode = aggregation or self.image_aggregation
        if mode == "max":
            return patch_scores.max(dim=1).values
        if mode == "topk_mean":
            k = self.top_k_for(patch_scores.shape[1])
            return torch.topk(patch_scores, k=k, dim=1).values.mean(dim=1)
        raise ValueError(
            f"image aggregation must be 'topk_mean' or 'max', got {mode!r}"
        )

    def enable_frequency_features(
        self,
        freq_num_bands: int = 6,
        freq_use_phase: bool = True,
        freq_feature_dim: int = 32,
        score_gamma: float | None = None,
    ) -> None:
        """Attach the parallel Fourier frequency stream.

        Built lazily so its submodules land on CPU; call model.to(device) AFTER
        this (and after load_state_dict) or its buffers stay behind.
        """
        from models.frequency_features import FourierPatchFeatureExtractor

        self.use_frequency = True
        if score_gamma is not None:
            self.score_gamma = score_gamma

        self.freq_extractor = FourierPatchFeatureExtractor(
            num_freq_bands=freq_num_bands,
            use_phase=freq_use_phase,
            feature_dim=freq_feature_dim,
        )
        # The extractor pads/truncates its raw descriptor to feature_dim, so that
        # is the actual width the frequency scorer sees.
        freq_dim = self.freq_extractor.feature_dim
        self.freq_mahalanobis_scorer = MahalanobisScoring(
            feature_dim=freq_dim,
            regularization=self.mahalanobis_scorer.regularization,
            gamma=self.mahalanobis_scorer.gamma,
        )
        self.freq_normal_stats_tracker = NormalStatisticsTracker(
            feature_dim=freq_dim,
            buffer_size=self.normal_stats_tracker.buffer_size,
            update_frequency=self.normal_stats_tracker.update_frequency,
        )
