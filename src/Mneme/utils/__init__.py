# init for utils subpackage

from .utils_imputer import reduce_imputers
from .utils_standardscaler import reduce_std_scalers
from .utils_minmaxscaler import reduce_minmax_scalers
from .utils_maxabsscaler import reduce_maxabs_scalers
from .utils_robustscaler import reduce_robust_scalers
from .utils_ordinalencoder import reduce_ordinal_encoders
from .utils_onehotencoder import reduce_onehot_encoders
from .utils_labelencoder import reduce_label_encoders
from .utils_ import _copy_attr, set_logging_level
from .._mneme_logging import _Mneme_logger
from functools import partial, update_wrapper
set_log_level = partial(set_logging_level, _Mneme_logger) 
update_wrapper(set_log_level,set_logging_level)

__all__ = ['reduce_imputers',
           'reduce_std_scalers',
           'reduce_minmax_scalers',
           'reduce_maxabs_scalers',
           'reduce_robust_scalers',
           'reduce_ordinal_encoders',
           'reduce_onehot_encoders',
           'reduce_label_encoders',
           '_copy_attr',
           'set_log_level']