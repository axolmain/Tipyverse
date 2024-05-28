import functools
import polars as pl


def chainable(func):
    """
    Decorator to make dataframe methods chainable. 
    Ensures that methods decorated with @chainable either modify the dataframe in-place or 
    return a new instance of TidyDataFrame to continue the method chain.

    Args:
        func (callable): A TidyDataFrame method that manipulates the dataframe.

    Returns:
        callable: A wrapper function that calls `func` and handles the chaining appropriately.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        wrapper_result = func(self, *args, **kwargs)
        return self if wrapper_result is None else wrapper_result

    return wrapper


class TidyDataFrame(pl.DataFrame):
    """
    Extends the Polars DataFrame to provide a fluent interface similar to dplyr from R, allowing for easy
    chaining of data manipulation methods.

    Args:
        data (dict, DataFrame, etc.): Data to be loaded into the DataFrame.

    Attributes:
        grouped (bool): Indicates whether the DataFrame is currently grouped.
        group_columns (list): Columns that the DataFrame is grouped by.
    """

    def __init__(self, data):
        super().__init__(data)
        self.grouped = False
        self.group_columns = []

    @chainable
    def select(self, *columns):
        """
        Selects specific columns from the DataFrame.

        Args:
            *columns (str): Column names to select.

        Returns:
            TidyDataFrame: A new DataFrame with only the specified columns.
        """
        return TidyDataFrame(super().select(columns))

    @chainable
    def filter(self, **conditions):
        """
        Filters rows based on specified conditions.

        Args:
            **conditions (dict): Conditions used to filter rows, where keys are column names and values are the conditions.

        Returns:
            TidyDataFrame: A new DataFrame containing only rows that meet the conditions.
        """
        exprs = [pl.col(k) == v if isinstance(v, (int, float, str)) else v for k, v in conditions.items()]
        combined_expr = functools.reduce(lambda a, b: a & b, exprs)
        return TidyDataFrame(super().filter(combined_expr))

    @chainable
    def arrange(self, *columns, ascending=True):
        """
        Sorts the DataFrame by specified columns.

        Args:
            *columns (str): Columns to sort by.
            ascending (bool): Sort direction. True for ascending, False for descending.

        Returns:
            TidyDataFrame: A new DataFrame sorted by the specified columns.
        """
        return TidyDataFrame(super().sort(columns, descending=not ascending))

    @chainable
    def mutate(self, **kwargs):
        """
        Adds new columns or modifies existing ones based on expressions provided.

        Args:
            **kwargs (dict): Mapping of new column names to their expressions.

        Returns:
            TidyDataFrame: A DataFrame with the modified or new columns.
        """
        mutations = [pl.col(expr).alias(new_col) if isinstance(expr, str) else expr.alias(new_col) for new_col, expr in
                     kwargs.items()]
        return TidyDataFrame(self.with_columns(mutations))

    @chainable
    def group_by(self, *columns):
        """
        Groups the DataFrame by specified columns.

        Args:
            *columns (str): Columns to group by.

        Returns:
            GroupBy object to be used for further operations like aggregation.
        """
        self.grouped = True
        self.group_columns = columns
        return self.groupby(columns)

    @chainable
    def summarize(self, groupby_columns, **kwargs):
        """
        Aggregates grouped data by applying functions specified in kwargs. Requires a GroupBy object.

        Args:
            groupby_obj (GroupBy): The GroupBy object to aggregate.
            **kwargs (dict): Mapping of output column names to aggregation functions.

        Raises:
            ValueError: If called before grouping the DataFrame.

        Returns:
            TidyDataFrame: A new DataFrame with aggregated data.
        """
        if not self.grouped:
            raise ValueError("summarize() can only be called after group_by()")
        exprs = [pl.col(col).sum().alias(name) for name, col in kwargs.items()]
        summarize_result = super().groupby(groupby_columns).agg(exprs)
        self.grouped = False
        return TidyDataFrame(summarize_result)


# Example usage
if __name__ == "__main__":
    df = TidyDataFrame({
        "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie"],
        "island": ["Torgersen", "Biscoe", "Dream", "Dream"],
        "body_mass_g": [3750, 3800, 5000, 3700]
    })

    # Example chain of operations
    result = (df
              .filter(species="Adelie")
              .mutate(body_mass_kg=pl.col("body_mass_g") / 1000)
              .select("species", "body_mass_kg")
              .arrange("body_mass_kg", ascending=False))

    print(result)
