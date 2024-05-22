from automodeldocs.chat.cache import simple_cache
from automodeldocs.response.formatted import FormattedOpenAIResponse


def test_cache():
    with simple_cache() as cache:
        cache.add_item(
            [{"role": "123", "content": "456"}, {"role": "123", "content": "567"}],
            [FormattedOpenAIResponse("system", "result")],
        )
    with simple_cache() as cache:
        res = cache.try_retrieve(
            [{"role": "123", "content": "456"}, {"role": "123", "content": "567"}]
        )
    assert res == [("system", "result")]
