"""
Audio Stitcher for Step Audio EditX TTS
Handles seamless stitching of audio chunks with crossfading
"""

import torch
from typing import List, Tuple


class AudioStitcher:
    """
    Stitch multiple audio chunks together with crossfading to eliminate clicks/pops.
    """

    # Default crossfade duration in milliseconds
    DEFAULT_CROSSFADE_MS = 50

    def __init__(self, sample_rate: int = 24000, crossfade_ms: int = None):
        """
        Initialize audio stitcher.

        Args:
            sample_rate: Audio sample rate in Hz (default: 24000 for Step Audio)
            crossfade_ms: Crossfade duration in milliseconds (default: 50ms)
        """
        self.sample_rate = sample_rate
        self.crossfade_ms = crossfade_ms or self.DEFAULT_CROSSFADE_MS
        self.crossfade_samples = int((self.crossfade_ms / 1000.0) * self.sample_rate)

    def stitch_chunks(
        self,
        audio_chunks: List[torch.Tensor],
        sample_rate: int = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Stitch multiple audio chunks together with crossfading.

        Args:
            audio_chunks: List of audio tensors (each can be 1D, 2D, or 3D)
            sample_rate: Sample rate (if None, uses instance sample_rate)

        Returns:
            (stitched_audio, sample_rate) tuple
        """
        if not audio_chunks:
            raise ValueError("No audio chunks to stitch")

        # Use provided sample rate or instance default
        sr = sample_rate or self.sample_rate

        # Update crossfade samples if sample rate changed
        if sample_rate and sample_rate != self.sample_rate:
            crossfade_samples = int((self.crossfade_ms / 1000.0) * sample_rate)
        else:
            crossfade_samples = self.crossfade_samples

        print(f"[StepAudio] 🔗 Stitching {len(audio_chunks)} audio chunks")
        print(f"[StepAudio]    Crossfade: {self.crossfade_ms}ms ({crossfade_samples} samples)")

        # Handle single chunk (no stitching needed)
        if len(audio_chunks) == 1:
            return audio_chunks[0], sr

        # Normalize all chunks to 1D tensors for easier processing
        normalized_chunks = []
        for i, chunk in enumerate(audio_chunks):
            norm_chunk = self._normalize_to_1d(chunk)
            normalized_chunks.append(norm_chunk)
            print(f"[StepAudio]      Chunk {i+1}: {norm_chunk.shape[0]} samples ({norm_chunk.shape[0] / sr:.2f}s)")

        # Stitch chunks with crossfading
        stitched = self._stitch_with_crossfade(normalized_chunks, crossfade_samples)

        print(f"[StepAudio]    ✓ Stitched audio: {stitched.shape[0]} samples ({stitched.shape[0] / sr:.2f}s)")

        return stitched, sr

    @staticmethod
    def _normalize_to_1d(audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio tensor to 1D shape.

        Args:
            audio_tensor: Input tensor (1D, 2D, or 3D)

        Returns:
            1D tensor [samples]
        """
        if audio_tensor.dim() == 1:
            # Already 1D: [samples]
            return audio_tensor

        elif audio_tensor.dim() == 2:
            # 2D: [channels, samples] or [batch, samples]
            # Take first channel/batch
            return audio_tensor[0]

        elif audio_tensor.dim() == 3:
            # 3D: [batch, channels, samples]
            # Take first batch and first channel
            return audio_tensor[0, 0]

        else:
            raise ValueError(f"Unsupported audio tensor dimension: {audio_tensor.dim()}")

    def _stitch_with_crossfade(
        self,
        chunks: List[torch.Tensor],
        crossfade_samples: int
    ) -> torch.Tensor:
        """
        Stitch chunks together with linear crossfading.

        Args:
            chunks: List of 1D audio tensors
            crossfade_samples: Number of samples for crossfade

        Returns:
            Stitched 1D audio tensor
        """
        if len(chunks) == 1:
            return chunks[0]

        # Start with first chunk
        stitched = chunks[0].clone()

        for i in range(1, len(chunks)):
            prev_chunk = stitched
            curr_chunk = chunks[i]

            # Determine actual crossfade length (limited by shorter chunk)
            actual_crossfade = min(
                crossfade_samples,
                prev_chunk.shape[0],
                curr_chunk.shape[0]
            )

            if actual_crossfade > 0:
                # Create linear fade weights
                fade_out = torch.linspace(1.0, 0.0, actual_crossfade, device=prev_chunk.device)
                fade_in = torch.linspace(0.0, 1.0, actual_crossfade, device=curr_chunk.device)

                # Apply crossfade to overlapping region
                overlap_start = prev_chunk.shape[0] - actual_crossfade
                prev_chunk[overlap_start:] = (
                    prev_chunk[overlap_start:] * fade_out +
                    curr_chunk[:actual_crossfade] * fade_in
                )

                # Concatenate: previous chunk (with crossfaded tail) + rest of current chunk
                stitched = torch.cat([prev_chunk, curr_chunk[actual_crossfade:]], dim=0)

            else:
                # No crossfade possible (chunks too short), just concatenate
                stitched = torch.cat([prev_chunk, curr_chunk], dim=0)

        return stitched

    def stitch_with_metadata(
        self,
        chunks_with_meta: List[Tuple[torch.Tensor, dict]],
        sample_rate: int = None
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Stitch chunks with metadata tracking.

        Args:
            chunks_with_meta: List of (audio_tensor, metadata_dict) tuples
            sample_rate: Sample rate (if None, uses instance sample_rate)

        Returns:
            (stitched_audio, sample_rate, combined_metadata) tuple
        """
        # Extract audio chunks
        audio_chunks = [chunk for chunk, _ in chunks_with_meta]

        # Stitch audio
        stitched_audio, sr = self.stitch_chunks(audio_chunks, sample_rate)

        # Combine metadata
        combined_meta = {
            "num_chunks": len(chunks_with_meta),
            "total_duration_s": stitched_audio.shape[0] / sr,
            "chunk_metadata": [meta for _, meta in chunks_with_meta]
        }

        return stitched_audio, sr, combined_meta


def create_stitcher(sample_rate: int = 24000, crossfade_ms: int = 50) -> AudioStitcher:
    """
    Factory function to create an AudioStitcher instance.

    Args:
        sample_rate: Audio sample rate in Hz
        crossfade_ms: Crossfade duration in milliseconds

    Returns:
        Configured AudioStitcher instance
    """
    return AudioStitcher(sample_rate=sample_rate, crossfade_ms=crossfade_ms)
