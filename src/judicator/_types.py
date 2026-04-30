from enum import Enum


class JudgeType(Enum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    BINARY = "binary"
    UNKNOWN = "unknown"
