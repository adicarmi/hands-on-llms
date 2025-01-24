import logging
from copy import deepcopy
from typing import Callable, List, Optional

from transformers import PreTrainedTokenizer
from unstructured.documents.elements import Element, NarrativeText, Text
from chonkie import SemanticChunker
from streaming_pipeline.base import SingletonMeta

logger = logging.getLogger(__name__)


def chunk_by_prefixed_attention_windows(
    text: str,
    tokenizer: PreTrainedTokenizer,
    buffer: int = 2,
    max_input_size: Optional[int] = None,
    split_function: Callable[[str], List[str]] = lambda text: text.split(" "),
    chunk_separator: str = " ",
    prefix: str = "",
) -> List[str]:
    """Splits a string of text into chunks that will fit into a model's attention
    window, with an optional fixed prefix added to each chunk.

    Parameters
    ----------
    text: The raw input text for the model
    tokenizer: The transformers tokenizer for the model
    buffer: Indicates the number of tokens to leave as a buffer for the attention window. This
        is to account for special tokens like [CLS] that can appear at the beginning or
        end of an input sequence.
    max_input_size: The size of the attention window for the model. If not specified, will
        use the model_max_length attribute on the tokenizer object.
    split_function: The function used to split the text into chunks to consider for adding to the
        attention window.
    chunk_separator: The string used to concat adjacent chunks when reconstructing the text
    prefix: A fixed string to prepend to each chunk.
    """
    max_input_size = tokenizer.model_max_length if max_input_size is None else max_input_size
    prefix_tokens = tokenizer.tokenize(prefix)
    prefix_token_count = len(prefix_tokens)

    if buffer < 0 or buffer >= max_input_size:
        raise ValueError(
            f"buffer is set to {buffer}. Must be greater than zero and smaller than "
            f"max_input_size, which is {max_input_size}.",
        )

    max_chunk_size = max_input_size - buffer - prefix_token_count

    if max_chunk_size <= 0:
        raise ValueError(
            f"The combination of buffer ({buffer}) and prefix token count ({prefix_token_count}) "
            f"exceeds or equals max_input_size ({max_input_size}). Reduce the buffer or prefix length."
        )

    split_text: List[str] = split_function(text)
    num_splits = len(split_text)

    chunks: List[str] = []
    chunk_text = ""
    chunk_size = 0

    for i, segment in enumerate(split_text):
        tokens = tokenizer.tokenize(segment)
        num_tokens = len(tokens)
        if num_tokens > max_chunk_size:
            raise ValueError(
                f"The number of tokens in the segment is {num_tokens}. "
                f"The maximum number of tokens is {max_chunk_size}. "
                "Consider using a different split_function to reduce the size "
                "of the segments under consideration. The text that caused the "
                f"error is: \n\n{segment}",
            )

        if chunk_size + num_tokens > max_chunk_size:
            chunks.append(prefix + chunk_text + chunk_separator.strip())
            chunk_text = ""
            chunk_size = 0

        # NOTE(robinson) - To avoid the separator appearing at the beginning of the string
        if chunk_size > 0:
            chunk_text += chunk_separator
        chunk_text += segment
        chunk_size += num_tokens

        if i == (num_splits - 1) and len(chunk_text) > 0:
            chunks.append(prefix + chunk_text)

    return chunks



class ChunkingSingleton(metaclass=SingletonMeta):
    """
    Singleton class for semantic text chunking using the SemanticChunker.

    This class ensures that a single instance of the SemanticChunker is used across the application.
    """

    def __init__(self):
        """
        Initializes the ChunkingSingleton with a default SemanticChunker instance.
        """
        self.chunker = SemanticChunker(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Pretrained model for sentence embeddings
            threshold=0.5,  # Similarity threshold for determining chunk boundaries
            chunk_size=384,  # Maximum number of tokens per chunk
            min_sentences=1  # Minimum number of sentences per chunk
        )

    def chunk_by_semantic(
        self,
        text: str,
        max_input_size: int,
        tokenizer: PreTrainedTokenizer,
        prefix: str = ""
    ) -> List[str]:
        """
        Chunk text into semantically meaningful chunks, ensuring each chunk fits within a specified token limit.

        Args:
            text (str): The text to be chunked.
            max_input_size (int): Maximum number of tokens allowed in each chunk (including the prefix).
            tokenizer (PreTrainedTokenizer): Tokenizer used to calculate token counts.
            prefix (str): Fixed string to prepend to each chunk.

        Returns:
            List[str]: A list of text chunks with the prefix prepended, ensuring token counts fit within max_input_size.

        Raises:
            ValueError: If the tokenized prefix exceeds max_input_size.
        """
        # Tokenize the prefix to calculate its token count
        prefix_tokens = tokenizer.tokenize(prefix)
        prefix_token_count = len(prefix_tokens)

        # Validate that the prefix length is less than the maximum input size
        if prefix_token_count >= max_input_size:
            raise ValueError(
                f"The prefix is too long ({prefix_token_count} tokens) to fit within the max_input_size ({max_input_size})."
            )

        # Adjust the maximum chunk size to account for the prefix length
        max_chunk_size = max_input_size - prefix_token_count

        # Set the chunk size for the SemanticChunker
        self.chunker.chunk_size = max_chunk_size

        # Perform semantic chunking
        raw_chunks = self.chunker.chunk(text)

        # Append the prefix to each chunk and return the list of chunks
        return [f"{prefix}{chunk.text}" for chunk in raw_chunks]

