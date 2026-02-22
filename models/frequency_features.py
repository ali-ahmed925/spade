"""Frequency-domain feature extraction for anomaly detection."""

import torch
import torch.nn as nn
import numpy as np
import cv2


class FourierPatchFeatureExtractor(nn.Module):
    """
    Extract frequency-domain features from 14×14 image patches.
    Designed for BLIP-2 ViT-B/14 patches.
    """
    
    def __init__(
        self,
        num_freq_bands: int = 6,
        use_phase: bool = True,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.use_phase = use_phase
        self.feature_dim = feature_dim
        
        # Compute feature size based on extraction method
        self._feature_size = self._compute_feature_size()
    
    def _compute_feature_size(self):
        """Compute the size of extracted feature vector."""
        size = 0
        # Per-band features (energy + mean) × num_bands
        size += self.num_freq_bands * 2
        # DC component
        size += 1
        # High-freq stats (3)
        size += 3
        # Mid-freq stats (3)
        size += 3
        # Phase (2) if enabled
        if self.use_phase:
            size += 2
        # Directional variance (1)
        size += 1
        # Edge density (1)
        size += 1
        return size
    
    def extract_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Extract frequency features from batch of patches.
        
        Args:
            patches: (N, 14, 14, 3) uint8 numpy array [0, 255]
        
        Returns:
            features: (N, feature_dim) float32 array
        """
        N = patches.shape[0]
        all_features = []
        
        for i in range(N):
            feat = self._extract_single(patches[i])
            all_features.append(feat)
        
        features = np.array(all_features, dtype=np.float32)  # (N, _feature_size)
        
        # Pad or truncate to target dimension
        if features.shape[1] < self.feature_dim:
            padding = np.zeros((N, self.feature_dim - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > self.feature_dim:
            features = features[:, :self.feature_dim]
        
        return features
    
    def _extract_single(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract frequency features from a single 14×14 patch.
        
        Args:
            patch: (14, 14, 3) uint8 array
        
        Returns:
            features: (_feature_size,) float32 array
        """
        # Convert to grayscale
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = patch.astype(np.float32)
        
        H, W = gray.shape  # 14, 14
        
        # 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)
        
        # Radial frequency coordinates
        center_y, center_x = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_r = np.sqrt(center_x**2 + center_y**2)
        
        features = []
        
        # 1. Radial frequency bands
        for i in range(self.num_freq_bands):
            r_inner = (i / self.num_freq_bands) * max_r
            r_outer = ((i + 1) / self.num_freq_bands) * max_r
            band_mask = (r >= r_inner) & (r < r_outer)
            
            band_mag = magnitude[band_mask]
            if len(band_mag) > 0:
                features.append(np.sum(band_mag**2))  # Energy
                features.append(np.mean(band_mag))     # Mean
            else:
                features.extend([0.0, 0.0])
        
        # 2. DC component
        features.append(magnitude[center_y, center_x])
        
        # 3. High-frequency stats
        high_freq_mask = r > (0.65 * max_r)
        if np.any(high_freq_mask):
            hf_mag = magnitude[high_freq_mask]
            features.append(np.sum(hf_mag**2))
            features.append(np.max(hf_mag))
            features.append(np.mean(hf_mag))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 4. Mid-frequency stats
        mid_freq_mask = (r > 0.25 * max_r) & (r < 0.65 * max_r)
        if np.any(mid_freq_mask):
            mf_mag = magnitude[mid_freq_mask]
            features.append(np.sum(mf_mag**2))
            features.append(np.std(mf_mag))
            
            # Spectral flatness
            geo_mean = np.exp(np.mean(np.log(mf_mag + 1e-8)))
            arith_mean = np.mean(mf_mag)
            features.append(geo_mean / (arith_mean + 1e-8))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 5. Phase information
        if self.use_phase and np.any(mid_freq_mask):
            mf_phase = phase[mid_freq_mask]
            features.append(np.var(mf_phase))
            features.append(np.mean(np.abs(mf_phase)))
        elif self.use_phase:
            features.extend([0.0, 0.0])
        
        # 6. Directional variance
        num_sectors = 4
        angles = np.arctan2(y - center_y, x - center_x)
        sector_energies = []
        
        for i in range(num_sectors):
            angle_start = -np.pi + i * (np.pi / num_sectors)
            angle_end = -np.pi + (i + 1) * (np.pi / num_sectors)
            sector_mask = (angles >= angle_start) & (angles < angle_end) & mid_freq_mask
            
            if np.any(sector_mask):
                sector_energies.append(np.sum(magnitude[sector_mask]**2))
            else:
                sector_energies.append(0.0)
        
        features.append(np.var(sector_energies))
        
        # 7. Edge density
        fft_highpass = fft_shift.copy()
        low_freq_mask = r <= (0.3 * max_r)
        fft_highpass[low_freq_mask] = 0
        
        hf_spatial = np.fft.ifft2(np.fft.ifftshift(fft_highpass))
        hf_magnitude = np.abs(hf_spatial)
        
        if hf_magnitude.size > 0:
            edge_threshold = np.percentile(hf_magnitude, 85)
            edge_density = np.sum(hf_magnitude > edge_threshold) / (H * W)
        else:
            edge_density = 0.0
        
        features.append(edge_density)
        
        return np.array(features, dtype=np.float32)
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        PyTorch forward pass.
        
        Args:
            patches: (N, 14, 14, 3) tensor [0, 255] uint8
        
        Returns:
            features: (N, feature_dim) tensor
        """
        # Convert to numpy
        if patches.is_cuda:
            patches_np = patches.cpu().numpy()
        else:
            patches_np = patches.numpy()
        
        # Ensure uint8
        patches_np = patches_np.astype(np.uint8)
        
        # Extract features
        features_np = self.extract_batch(patches_np)
        
        # Convert back to torch
        features = torch.from_numpy(features_np).to(patches.device)
        
        return features

