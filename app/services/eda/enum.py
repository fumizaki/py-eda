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


class CorrelationMethod(str, Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class OutlierMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"


class FillStrategyMethod(str, Enum):
    MEAN = 'mean'
    MODE = 'mode'
    MEDIAN = 'median'
    ZERO = 'zero'
    FFILL = 'ffill'
    BFILL = 'bfill'
