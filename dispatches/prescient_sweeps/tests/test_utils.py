from dispatches.prescient_sweeps.utils import FlattenedIndexMapper
from itertools import product, zip_longest
import pytest

class TestFlattenedIndexMapper:

    @pytest.fixture
    def data(self):
        return { "foo" : list(range(1,5)),
                 "bar" : ["a", "b", "c"],
                 "baz" : [str(i) for i in range(5,50)],
               }

    def test_lengths_points(self, data):
        m = FlattenedIndexMapper(data)
        assert m._lengths["foo"] == 4
        assert m._lengths["bar"] == 3
        assert m._lengths["baz"] == 50-5
        assert m.number_of_points == 4*3*(50-5)

    def test_indices(self, data):
        m = FlattenedIndexMapper(data)
        for idx, (foo, bar, baz) in enumerate(product(*data.values())):
            point = m.get_point(idx)
            assert point["foo"] == foo
            assert point["bar"] == bar
            assert point["baz"] == baz

            point = m(idx)
            assert point["foo"] == foo
            assert point["bar"] == bar
            assert point["baz"] == baz

        for (idx, point), (foo, bar, baz) in zip_longest(m.all_points_generator(), product(*data.values())):
            assert point["foo"] == foo
            assert point["bar"] == bar
            assert point["baz"] == baz
