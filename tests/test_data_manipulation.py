import polars as pl
import pytest
from src.tipyverse import select, filter, mutate, arrange, summarize, group_by


@pytest.fixture
def penguins():
    return pl.DataFrame({
        'island': ['Biscoe', 'Dream', 'Biscoe', 'Dream', 'Torgersen', 'Biscoe'],
        'bill_depth_mm': [18.7, 17.4, 18.0, 17.6, 18.5, 19.3]
    })


def test_select(penguins):
    result = select(penguins, 'island', 'bill_depth_mm')
    assert result.columns == ['island', 'bill_depth_mm']


def test_filter(penguins):
    result = filter(penguins, island="Biscoe")
    assert all(result['island'] == 'Biscoe')


def test_arrange(penguins):
    result = arrange(penguins, 'bill_depth_mm', ascending=False)
    assert result['bill_depth_mm'].to_list()[0] >= result['bill_depth_mm'].to_list()[-1]


def test_mutate(penguins):
    result = mutate(penguins, bill_depth_cm=pl.col('bill_depth_mm') / 10)
    assert 'bill_depth_cm' in result.columns
    assert all(result['bill_depth_cm'] == result['bill_depth_mm'] / 10)


def test_summarize(penguins):
    result = summarize(penguins, bill_depth_mean=pl.col('bill_depth_mm').mean())
    assert 'bill_depth_mean' in result.columns
