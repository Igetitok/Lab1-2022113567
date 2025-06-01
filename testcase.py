import pytest
from Grapy import query_bridge_words


# 定义测试用的图数据（基于用户提供的文本）
@pytest.fixture
def custom_graph():
    # 原始文本处理后得到的单词列表：
    words = [
        "the", "scientist", "carefully", "analyzed", "the", "data",
        "wrote", "a", "detailed", "report", "and", "shared", "the",
        "report", "with", "the", "team", "but", "the", "team",
        "requested", "more", "data", "so", "the", "scientist",
        "analyzed", "it", "again"
    ]

    # 手动构建图结构（模拟黑盒输入）
    graph = {
        "the": {
            "scientist": 1,
            "data": 1,
            "report": 2,
            "team": 2,
            "scientist": 1  # 注意：最后的 "the scientist" 对应一次
        },
        "scientist": {"carefully": 1, "analyzed": 1},
        "carefully": {"analyzed": 1},
        "analyzed": {"the": 1, "it": 1},
        "data": {"wrote": 1, "so": 1},
        "wrote": {"a": 1},
        "a": {"detailed": 1},
        "detailed": {"report": 1},
        "report": {"and": 1, "with": 1},
        "and": {"shared": 1},
        "shared": {"the": 1},
        "with": {"the": 1},
        "team": {"but": 1, "requested": 1},
        "but": {"the": 1},
        "requested": {"more": 1},
        "more": {"data": 1},
        "so": {"the": 1},
        "it": {"again": 1},
        "again": {}
    }
    return graph

def test_bridge_words_analyzed_to_data(custom_graph):
    # 查询 "analyzed" → "data" 的桥接词
    result = query_bridge_words(custom_graph, "analyzed", "data")
    # 桥接词路径: analyzed → the → data
    assert "The bridge word from analyzed to data is: the." == result


# 测试用例
def test_bridge_words_scientist_to_data(custom_graph):
    # 查询 "scientist" → "data" 的桥接词
    result = query_bridge_words(custom_graph, "scientist", "data")
    # 可能的路径: scientist → carefully → analyzed → the → data
    # 但桥接词需直接连接，即 scientist → X → data，此处应无桥接词
    assert "No bridge words from scientist to data!" in result


def test_bridge_words_the_to_team(custom_graph):
    # 查询 "the" → "team" 的桥接词
    result = query_bridge_words(custom_graph, "the", "team")
    # 直接路径: the → team（无需桥接词）
    assert "No bridge words from the to team!" in result


def test_bridge_words_analyzed_to_it(custom_graph):
    # 查询 "analyzed" → "it" 的桥接词
    result = query_bridge_words(custom_graph, "analyzed", "it")
    # 直接路径: analyzed → it（无需桥接词）
    assert "No bridge words from analyzed to it!" in result


def test_bridge_words_report_to_with(custom_graph):
    # 查询 "report" → "with" 的桥接词
    result = query_bridge_words(custom_graph, "report", "with")
    # 直接路径: report → with（无需桥接词）
    assert "No bridge words from report to with!" in result


def test_word_not_in_graph(custom_graph):
    # 查询不存在于图中的单词
    result = query_bridge_words(custom_graph, "apple", "banana")
    assert "No word1 or word2 in the graph!" in result