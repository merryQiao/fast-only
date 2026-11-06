from tools.train_slowfast_expanded import SnapshotLRU


def test_snapshots_are_bounded_and_recomputed_after_eviction_or_model_change():
    cache=SnapshotLRU(2); calls=[]
    def factory(key):
        calls.append(key)
        return object()
    first=cache.get('a',lambda:factory('a'))
    assert cache.get('a',lambda:factory('a')) is first
    cache.get('b',lambda:factory('b'))
    cache.get('c',lambda:factory('c'))
    assert len(cache)==2 and calls==['a','b','c']
    assert cache.get('a',lambda:factory('a')) is not first
    cache.clear()
    assert len(cache)==0
    assert cache.get('a',lambda:factory('a')) is not first


def test_snapshot_cache_never_grows_with_full_cohort():
    cache=SnapshotLRU(2)
    for token in range(324):
        assert cache.get(token,lambda:token)==token
        assert len(cache)<=2
