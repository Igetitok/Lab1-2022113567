import pytest
from collections import defaultdict
from Grapy import dijkstra, calc_shortest_path, calc_all_shortest_paths_from_word


@pytest.fixture
def test_graphs():
    """提供各种测试图结构的fixture"""
    graphs = {}

    # 简单线性图 A → B → C → D
    graphs['simple'] = defaultdict(lambda: defaultdict(int))
    graphs['simple']['A']['B'] = 1
    graphs['simple']['B']['C'] = 2
    graphs['simple']['C']['D'] = 3

    # 带权重的多路径图
    graphs['weighted'] = defaultdict(lambda: defaultdict(int))
    graphs['weighted']['A']['B'] = 1
    graphs['weighted']['A']['C'] = 4
    graphs['weighted']['B']['C'] = 2
    graphs['weighted']['C']['D'] = 1

    # 带环图 A → B → C → A → D
    graphs['cyclic'] = defaultdict(lambda: defaultdict(int))
    graphs['cyclic']['A']['B'] = 1
    graphs['cyclic']['B']['C'] = 1
    graphs['cyclic']['C']['A'] = 1
    graphs['cyclic']['C']['D'] = 1

    # 不连通图
    graphs['disconnected'] = defaultdict(lambda: defaultdict(int))
    graphs['disconnected']['A']['B'] = 1
    graphs['disconnected']['C']['D'] = 1

    return graphs


# 基本路径1：起点或终点不在图中
def test_dijkstra_start_not_in_graph(test_graphs):
    """测试起点不在图中的情况"""
    path, dist = dijkstra(test_graphs['simple'], 'X', 'A')
    assert path == []
    assert dist == float('inf')


def test_dijkstra_end_not_in_graph(test_graphs):
    """测试终点不在图中的情况"""
    path, dist = dijkstra(test_graphs['simple'], 'A', 'X')
    assert path == []
    assert dist == float('inf')


# 基本路径2：起点等于终点
def test_dijkstra_start_equals_end(test_graphs):
    """测试起点和终点相同的情况"""
    path, dist = dijkstra(test_graphs['simple'], 'A', 'A')
    assert path == ['A']
    assert dist == 0


# 基本路径3：起点没有出边（孤立节点）
def test_dijkstra_start_no_out_edges():
    """测试起点没有出边的情况"""
    graph = defaultdict(lambda: defaultdict(int))
    graph['A'] = {}  # 无出边
    graph['B']['C'] = 1
    path, dist = dijkstra(graph, 'A', 'C')
    assert path == []
    assert dist == float('inf')


# 基本路径4：终点不可达
def test_dijkstra_unreachable_node(test_graphs):
    """测试终点不可达的情况"""
    path, dist = dijkstra(test_graphs['disconnected'], 'A', 'D')
    assert path == []
    assert dist == float('inf')


# 基本路径5：直接邻居节点
def test_dijkstra_direct_neighbor(test_graphs):
    """测试直接相邻节点的情况"""
    path, dist = dijkstra(test_graphs['simple'], 'A', 'B')
    assert path == ['A', 'B']
    assert dist == 1


# 基本路径6：多步路径
def test_dijkstra_multi_step_path(test_graphs):
    """测试需要多步的路径"""
    path, dist = dijkstra(test_graphs['simple'], 'A', 'D')
    assert path == ['A', 'B', 'C', 'D']
    assert dist == 6


# 基本路径7：权重影响路径选择
def test_dijkstra_weighted_path_selection(test_graphs):
    """测试权重影响路径选择的情况"""
    path, dist = dijkstra(test_graphs['weighted'], 'A', 'C')
    assert path == ['A', 'B', 'C']
    assert dist == 3


# 基本路径8：带环图的路径
def test_dijkstra_with_cycle(test_graphs):
    """测试带环图的路径查找"""
    path, dist = dijkstra(test_graphs['cyclic'], 'A', 'D')
    assert path == ['A', 'B', 'C', 'D']
    assert dist == 3


# 基本路径9：路径重建失败（正确实现中不应发生）
def test_dijkstra_path_reconstruction_failure():
    """测试路径重建失败的情况（正确实现中应能处理）"""
    graph = defaultdict(lambda: defaultdict(int))
    graph['A']['B'] = 1
    graph['B']['C'] = 1
    path, dist = dijkstra(graph, 'A', 'C')
    assert path == ['A', 'B', 'C']  # 正确实现中应该能工作
    assert dist == 2


# 测试calc_shortest_path包装函数
def test_calc_shortest_path_valid(test_graphs):
    """测试包装函数处理有效路径的情况"""
    result = calc_shortest_path(test_graphs['simple'], 'A', 'B')
    assert result == "Shortest path: A→B, length: 1"


def test_calc_shortest_path_invalid(test_graphs):
    """测试包装函数处理无效路径的情况"""
    result = calc_shortest_path(test_graphs['simple'], 'A', 'X')
    assert result == "No path between A and X"


# 测试calc_all_shortest_paths_from_word函数
def test_calc_all_shortest_paths(test_graphs):
    """测试从某个节点出发的所有最短路径"""
    result = calc_all_shortest_paths_from_word(test_graphs['simple'], 'A')
    expected = """To B: A→B (length: 1)
To C: A→B→C (length: 3)
To D: A→B→C→D (length: 6)"""
    assert result == expected


def test_calc_all_shortest_paths_invalid_word(test_graphs):
    """测试使用无效起始词的情况"""
    result = calc_all_shortest_paths_from_word(test_graphs['simple'], 'X')
    assert result == "Word 'x' not in graph!"