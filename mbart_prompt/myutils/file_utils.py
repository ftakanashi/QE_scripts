#!/usr/bin/env python

from dataclasses import dataclass
from enum import Enum
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, Any

class ExplicitEnum(Enum):

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"