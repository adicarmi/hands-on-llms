import datetime
from typing import List, Tuple
from copy import deepcopy
from typing import Callable, List, Optional

from transformers import PreTrainedTokenizer

from unstructured.documents.elements import Element, NarrativeText, Text


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

def read_requirements(file_path: str) -> List[str]:
    """
    Reads a file containing a list of requirements and returns them as a list of strings.

    Args:
        file_path (str): The path to the file containing the requirements.

    Returns:
        List[str]: A list of requirements as strings.
    """

    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    return requirements


def split_time_range_into_intervals(
    from_datetime: datetime.datetime, to_datetime: datetime.datetime, n: int
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Splits a time range [from_datetime, to_datetime] into N equal intervals.

    Args:
        from_datetime (datetime): The starting datetime object.
        to_datetime (datetime): The ending datetime object.
        n (int): The number of intervals.

    Returns:
        List of tuples: A list where each tuple contains the start and end datetime objects for each interval.
    """

    # Calculate total duration between from_datetime and to_datetime.
    total_duration = to_datetime - from_datetime

    # Calculate the length of each interval.
    interval_length = total_duration / n

    # Generate the interval.
    intervals = []
    for i in range(n):
        interval_start = from_datetime + (i * interval_length)
        interval_end = from_datetime + ((i + 1) * interval_length)
        if i + 1 != n:
            # Subtract 1 microsecond from the end of each interval to avoid overlapping.
            interval_end = interval_end - datetime.timedelta(minutes=1)

        intervals.append((interval_start, interval_end))

    return intervals
