import pytest
from Grapy import calc_shortest_path


# 根据指定文本构建图
@pytest.fixture
def custom_graph():
    # 处理后的单词列表
    words = [
        "the", "scientist", "carefully", "analyzed", "the", "data",
        "wrote", "a", "detailed", "report", "and", "shared", "the",
        "report", "with", "the", "team", "but", "the", "team",
        "requested", "more", "data", "so", "the", "scientist",
        "analyzed", "it", "again"
    ]

    # 构建图结构（边权重为单词连续出现的次数）
    graph = {
        "the": {
            "scientist": 2,  # "the scientist" 出现2次
            "data": 2,  # "the data" 出现2次
            "report": 2,  # "the report" 出现2次
            "team": 2  # "the team" 出现2次
        },
        "scientist": {
            "carefully": 1,
            "analyzed": 2  # "scientist analyzed" 出现2次
        },
        "carefully": {"analyzed": 1},
        "analyzed": {
            "the": 1,
            "it": 1
        },
        "data": {
            "wrote": 1,
            "so": 1
        },
        "wrote": {"a": 1},
        "a": {"detailed": 1},
        "detailed": {"report": 1},
        "report": {
            "and": 1,
            "with": 1
        },
        "and": {"shared": 1},
        "shared": {"the": 1},
        "with": {"the": 1},
        "team": {
            "but": 1,
            "requested": 1
        },
        "but": {"the": 1},
        "requested": {"more": 1},
        "more": {"data": 1},
        "so": {"the": 1},
        "it": {"again": 1},
        "again": {}
    }
    return graph


# 测试用例
def test_shortest_path_scientist_to_data(custom_graph):
    # 路径：scientist → analyzed → the → data（权重 2+1+2=5）
    result = calc_shortest_path(custom_graph, "scientist", "data")
    assert "scientist→analyzed→the→data" in result and "length: 5" in result


def test_shortest_path_the_to_report(custom_graph):
    # 直接路径：the → report（权重 2）
    result = calc_shortest_path(custom_graph, "the", "report")
    assert "the→report" in result and "length: 2" in result


def test_shortest_path_team_to_more(custom_graph):
    # 路径：team → requested → more（权重 1+1=2）
    result = calc_shortest_path(custom_graph, "team", "more")
    assert "team→requested→more" in result and "length: 2" in result


def test_no_path_exists(custom_graph):
    # 无路径：again 无法到达其他节点
    result = calc_shortest_path(custom_graph, "again", "the")
    assert "No path between again and the" in result


def test_same_start_and_end(custom_graph):
    # 起点和终点相同
    result = calc_shortest_path(custom_graph, "report", "report")
    assert "Shortest path: report, length: 0" in result

def test_start_node_not_exist(custom_graph):
    """测试起点不存在的情况（基本路径6）"""
    result = calc_shortest_path(custom_graph, "apple", "report")
    assert "No path between apple and report" in result

def test_end_node_not_exist(custom_graph):
    """测试终点不存在的情况（基本路径7）"""
    result = calc_shortest_path(custom_graph, "the", "banana")
    assert "No path between the and banana" in result