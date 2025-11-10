"""
Longform Text Chunker for Step Audio EditX TTS
Handles smart sentence-based chunking for long-form text generation
"""

import re
from typing import List, Tuple


class LongformChunker:
    """
    Intelligently chunk long text at sentence boundaries with sliding window overlap.
    """

    # Approximate token estimation: 1 token ≈ 4 characters (conservative estimate)
    CHARS_PER_TOKEN = 4

    # Overlap ratio for sliding window (disabled for clean chunking)
    OVERLAP_RATIO = 0.0  # No overlap - each chunk is independent

    # Minimum chunk size in tokens (avoid tiny chunks)
    # Note: This can be overridden dynamically based on max_new_tokens
    MIN_CHUNK_TOKENS = 50  # Lowered from 256 to allow smaller chunks

    def __init__(self, max_tokens: int = 8192, tokenizer=None, enforce_minimum: bool = True):
        """
        Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            tokenizer: Optional tokenizer for accurate token counting
            enforce_minimum: Whether to enforce MIN_CHUNK_TOKENS (default True)
        """
        # Allow disabling minimum for very small max_new_tokens scenarios
        if enforce_minimum:
            self.max_tokens = max(self.MIN_CHUNK_TOKENS, max_tokens)
        else:
            self.max_tokens = max(10, max_tokens)  # At least 10 tokens
        self.max_chars = self.max_tokens * self.CHARS_PER_TOKEN
        self.overlap_chars = int(self.max_chars * self.OVERLAP_RATIO)
        self.tokenizer = tokenizer  # Store tokenizer for accurate counting

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using regex-based detection.

        Handles:
        - Standard punctuation (. ! ?)
        - Abbreviations (Mr. Dr. etc.)
        - Quotes and parentheses
        - Multiple punctuation marks

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Pattern to match sentence boundaries
        # Looks for sentence-ending punctuation followed by whitespace and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'

        # Split on pattern
        sentences = re.split(sentence_pattern, text)

        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # Handle edge case: if no sentences found, return the whole text
        if not sentences:
            return [text.strip()]

        return sentences

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Args:
            text: Input text

        Returns:
            Estimated token count (accurate if tokenizer provided, otherwise conservative estimate)
        """
        # Use actual tokenizer if available for accurate counting
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass  # Fall back to character-based estimate

        # Fallback: character-based estimate (conservative)
        return len(text) // self.CHARS_PER_TOKEN

    def needs_chunking(self, text: str) -> bool:
        """
        Check if text needs to be chunked.

        Chunks when text is long enough to benefit from chunking (>25% of max_tokens).
        This improves performance since long generations slow down (decreasing it/s).

        Args:
            text: Input text

        Returns:
            True if text should be chunked
        """
        estimated_tokens = self.estimate_tokens(text)
        # Chunk if text is longer than 25% of max_tokens
        # This ensures even medium-length texts get chunked for better performance
        min_chunk_threshold = self.max_tokens * 0.25
        return estimated_tokens > min_chunk_threshold

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk text at sentence boundaries with sliding window overlap.

        Args:
            text: Input text to chunk

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        # Check if chunking is needed
        if not self.needs_chunking(text):
            return [(text, 0, len(text))]

        print(f"[StepAudio] 📝 Chunking long text ({self.estimate_tokens(text)} tokens estimated)")
        print(f"[StepAudio]    Max tokens per chunk: {self.max_tokens}")
        print(f"[StepAudio]    Overlap: {int(self.OVERLAP_RATIO * 100)}%")

        # Split into sentences
        sentences = self.split_into_sentences(text)
        print(f"[StepAudio]    Found {len(sentences)} sentences")

        # Fallback: If only 1 sentence but text exceeds max_tokens, split by words
        if len(sentences) == 1 and self.estimate_tokens(text) > self.max_tokens:
            print(f"[StepAudio]    ⚠️  Single sentence exceeds max_tokens, splitting by words...")
            words = text.split()
            sentences = []
            current_sentence = []
            current_tokens = 0

            for word in words:
                word_tokens = self.estimate_tokens(word)
                if current_tokens + word_tokens > self.max_tokens and current_sentence:
                    # Save current sentence and start new one
                    sentences.append(' '.join(current_sentence))
                    current_sentence = [word]
                    current_tokens = word_tokens
                else:
                    current_sentence.append(word)
                    current_tokens += word_tokens

            # Add final sentence
            if current_sentence:
                sentences.append(' '.join(current_sentence))

            print(f"[StepAudio]    ✓ Split into {len(sentences)} word-based chunks")

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start_char = 0

        for i, sentence in enumerate(sentences):
            # Use accurate token counting if tokenizer available, otherwise use character count
            if self.tokenizer is not None:
                try:
                    sentence_length = len(self.tokenizer.encode(sentence))
                    chunk_text_temp = ' '.join(current_chunk + [sentence])
                    chunk_tokens = len(self.tokenizer.encode(chunk_text_temp))
                    would_exceed = chunk_tokens > self.max_tokens
                except Exception:
                    # Fall back to character-based chunking
                    sentence_length = len(sentence)
                    would_exceed = current_length + sentence_length > self.max_chars
            else:
                sentence_length = len(sentence)
                would_exceed = current_length + sentence_length > self.max_chars

            # Check if adding this sentence would exceed limit
            if would_exceed and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_end_char = chunk_start_char + len(chunk_text)
                chunks.append((chunk_text, chunk_start_char, chunk_end_char))

                # Calculate overlap for next chunk
                # Go back and include last few sentences for context
                overlap_sentences = []
                overlap_length = 0
                overlap_limit = self.max_tokens if self.tokenizer else self.overlap_chars

                for prev_sentence in reversed(current_chunk):
                    if self.tokenizer is not None:
                        try:
                            overlap_text = ' '.join(overlap_sentences + [prev_sentence])
                            overlap_tokens = len(self.tokenizer.encode(overlap_text))
                            if overlap_tokens <= int(self.max_tokens * self.OVERLAP_RATIO):
                                overlap_sentences.insert(0, prev_sentence)
                                overlap_length = overlap_tokens
                            else:
                                break
                        except Exception:
                            # Fall back to character-based overlap
                            if overlap_length + len(prev_sentence) <= self.overlap_chars:
                                overlap_sentences.insert(0, prev_sentence)
                                overlap_length += len(prev_sentence) + 1
                            else:
                                break
                    else:
                        if overlap_length + len(prev_sentence) <= self.overlap_chars:
                            overlap_sentences.insert(0, prev_sentence)
                            overlap_length += len(prev_sentence) + 1  # +1 for space
                        else:
                            break

                # Start new chunk with overlap
                current_chunk = overlap_sentences + [sentence]
                if self.tokenizer is not None:
                    try:
                        current_length = len(self.tokenizer.encode(' '.join(current_chunk)))
                    except Exception:
                        current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                else:
                    current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                chunk_start_char = chunk_end_char - overlap_length

            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                if self.tokenizer is not None:
                    try:
                        current_length = len(self.tokenizer.encode(' '.join(current_chunk)))
                    except Exception:
                        current_length += sentence_length + 1
                else:
                    current_length += sentence_length + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_end_char = len(text)
            chunks.append((chunk_text, chunk_start_char, chunk_end_char))

        print(f"[StepAudio]    Created {len(chunks)} chunks")

        # Validate chunks
        for i, (chunk_text, start, end) in enumerate(chunks):
            tokens = self.estimate_tokens(chunk_text)
            print(f"[StepAudio]      Chunk {i+1}/{len(chunks)}: {tokens} tokens (~{len(chunk_text)} chars)")

        return chunks

    def get_chunk_info(self, chunk_index: int, total_chunks: int) -> str:
        """
        Get human-readable chunk information for progress display.

        Args:
            chunk_index: Current chunk index (0-based)
            total_chunks: Total number of chunks

        Returns:
            Formatted string like "Chunk 2/5"
        """
        return f"Chunk {chunk_index + 1}/{total_chunks}"


def create_chunker(max_tokens: int = 8192, tokenizer=None, enforce_minimum: bool = True) -> LongformChunker:
    """
    Factory function to create a LongformChunker instance.

    Args:
        max_tokens: Maximum tokens per chunk
        tokenizer: Optional tokenizer for accurate token counting
        enforce_minimum: Whether to enforce MIN_CHUNK_TOKENS (default True)

    Returns:
        Configured LongformChunker instance
    """
    return LongformChunker(max_tokens=max_tokens, tokenizer=tokenizer, enforce_minimum=enforce_minimum)
