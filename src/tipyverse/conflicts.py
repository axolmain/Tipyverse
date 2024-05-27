import sys


def tipyverse_conflicts():
    # Detect conflicts between packages in the tipyverse and other loaded packages
    conflicts = set(sys.modules.keys()) & {'polars', 'numpy', 'seaborn', 'plotnine', 'toolz'}
    return conflicts
