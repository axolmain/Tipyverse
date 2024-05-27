import polars as pl


def select(df: pl.DataFrame, *columns):
    polars_columns = [pl.col(col) for col in columns]
    return df.select(polars_columns)


def filter(df: pl.DataFrame, **conditions):
    exprs = [(pl.col(k) == v) if isinstance(v, (int, float, str)) else v for k, v in conditions.items()]
    combined_expr = exprs[0]
    for expr in exprs[1:]:
        combined_expr = combined_expr & expr
    return df.filter(combined_expr)


def arrange(df: pl.DataFrame, *columns, ascending=True):
    return df.sort(list(columns), descending=not ascending)


def mutate(df: pl.DataFrame, **kwargs):
    for new_col, expr in kwargs.items():
        if isinstance(expr, str):
            expr = pl.col(expr)
        df = df.with_columns(expr.alias(new_col))
    return df


def summarize(df: pl.DataFrame, **kwargs):
    exprs = [func.alias(col) if not isinstance(func, str) else pl.col(func).alias(col) for col, func in kwargs.items()]
    return df.select(exprs)


def group_by(df: pl.DataFrame, *columns):
    return df.groupby(list(columns))


def summarize_grouped(df: pl.DataFrame, **kwargs):
    exprs = [func.alias(col) for col, func in kwargs.items()]
    return df.agg(exprs)
