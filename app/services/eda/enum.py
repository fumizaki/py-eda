from enum import Enum


class DescribeType(str, Enum):
    ALL = 'all'
    NUMBER = 'number'
    OBJECT = 'object'
    CATEGORY = 'category'
    BOOL = 'bool'    


class EncodingType(str, Enum):
    ONEHOT = "onehot"
    LABEL = "label"
    ORDINAL = "ordinal"


class ScalingMethod(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"


class CorrelationMethod(str, Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class DetectOutlierMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"

class TreatOutlierMethod(str, Enum):
    REMOVE = "remove"
    CLIP = "clip"
    REPLACE = "replace"


class ImputeMethod(str, Enum):
    MEAN = 'mean'
    MODE = 'mode'
    MEDIAN = 'median'
    FFILL = 'ffill' # Forward Fill (直前の値で補完)
    BFILL = 'bfill' # Backward Fill (直後の値で補完)
