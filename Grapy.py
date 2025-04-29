import re
import heapq
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


# 文本处理与图构建
def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    return words


def build_graph(words):
    graph = defaultdict(lambda: defaultdict(int))
    all_words = set(words)
    for word in all_words:
        graph[word]  # 确保所有单词作为节点存在
    for i in range(len(words) - 1):
        current, next_word = words[i], words[i + 1]
        graph[current][next_word] += 1
    return graph


# 桥接词查询
def get_bridge_words(graph, word1, word2):
    word1, word2 = word1.lower(), word2.lower()
    if word1 not in graph or word2 not in graph:
        return None
    bridges = []
    for candidate in graph[word1]:
        if candidate in graph and word2 in graph[candidate]:
            bridges.append(candidate)
    return bridges


def query_bridge_words(graph, word1, word2):
    bridges = get_bridge_words(graph, word1, word2)
    if bridges is None:
        return "No word1 or word2 in the graph!"
    if not bridges:
        return f"No bridge words from {word1} to {word2}!"
    if len(bridges) == 1:
        return f"The bridge word from {word1} to {word2} is: {bridges[0]}."
    else:
        return f"The bridge words from {word1} to {word2} are: {', '.join(bridges[:-1])} and {bridges[-1]}."


# 生成新文本
def generate_new_text(graph, text):
    words = re.sub(r'[^a-z]', ' ', text.lower()).split()
    new_text = []
    for i in range(len(words) - 1):
        new_text.append(words[i])
        bridges = get_bridge_words(graph, words[i], words[i + 1])
        if bridges:
            new_text.append(random.choice(bridges))
    new_text.append(words[-1])
    return ' '.join(new_text)


# 最短路径（自主实现Dijkstra算法）
def dijkstra(graph, start, end):
    if start not in graph or end not in graph:
        return [], float('inf')

    nodes = list(graph.keys())
    distances = {node: float('inf') for node in nodes}
    predecessors = {node: None for node in nodes}
    distances[start] = 0

    heap = []
    heapq.heappush(heap, (0, start))

    while heap:
        current_dist, current = heapq.heappop(heap)
        if current_dist > distances[current]:
            continue

        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current
                heapq.heappush(heap, (distance, neighbor))

    if distances[end] == float('inf'):
        return [], float('inf')

    # 回溯路径
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()

    if not path or path[0] != start:
        return [], float('inf')
    return path, distances[end]


def calc_shortest_path(graph, word1, word2):
    path, dist = dijkstra(graph, word1.lower(), word2.lower())
    if not path:
        return f"No path between {word1} and {word2}"
    return f"Shortest path: {'→'.join(path)}, length: {dist}"


def calc_all_shortest_paths_from_word(graph, word):
    word = word.lower()
    if word not in graph:
        return f"Word '{word}' not in graph!"

    results = []
    for target in sorted(graph.keys()):  # 按字母排序
        if target == word:
            continue
        path, dist = dijkstra(graph, word, target)
        if path:
            results.append(f"To {target}: {'→'.join(path)} (length: {dist})")
        else:
            results.append(f"To {target}: No path")

    if not results:
        return f"No other nodes reachable from {word}"
    return "\n".join(results)


def handle_shortest_path_input(graph):
    inputs = input("Enter word(s) (separate by space): ").lower().split()

    if len(inputs) == 1:
        print("\n" + calc_all_shortest_paths_from_word(graph, inputs[0]))
    elif len(inputs) == 2:
        print("\n" + calc_shortest_path(graph, inputs[0], inputs[1]))
    else:
        print("Please enter 1 or 2 words!")


# PageRank（自主实现）
def calculate_pagerank(graph, d=0.85, max_iter=100, tol=1e-6):
    nodes = list(graph.keys())
    if not nodes:
        return {}

    N = len(nodes)
    pr = {node: 1.0 / N for node in nodes}

    # 计算每个节点的总出权重（权重之和）
    total_out_weight = {node: sum(neighbors.values()) for node, neighbors in graph.items()}

    # 预处理入链
    in_links = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            in_links[neighbor].append(node)

    # 收集悬挂节点（总出权重为0的节点）
    dangling_nodes = [node for node in nodes if total_out_weight[node] == 0]

    for _ in range(max_iter):
        old_pr = pr.copy()
        dangling_sum = sum(old_pr[node] for node in dangling_nodes)
        dangling_contrib = d * dangling_sum / N  # 悬挂节点的贡献均分

        new_pr = {}
        for node in nodes:
            # 计算来自入链的贡献（考虑权重）
            in_contrib = 0.0
            for v in in_links[node]:
                if total_out_weight[v] > 0:
                    # 贡献 = 源节点的PR值 * (边权重 / 源节点总出权重)
                    in_contrib += old_pr[v] * (graph[v][node] / total_out_weight[v])
            # 总PR值 = 随机跳转 + 阻尼因子 * (入链贡献 + 悬挂贡献)
            new_pr[node] = (1 - d) / N + d * (in_contrib + dangling_contrib / d)  # 修正悬挂贡献计算

        # 检查收敛
        delta = sum(abs(new_pr[node] - old_pr[node]) for node in nodes)
        if delta < tol:
            break
        pr = new_pr

    # 归一化PR值（可选，根据NetworkX是否归一化）
    pr_sum = sum(pr.values())
    pr = {k: v / pr_sum for k, v in pr.items()}
    return pr


# 随机游走
def random_walk(graph):
    if not graph:
        return []
    nodes = list(graph.keys())
    current = random.choice(nodes)
    path = [current]
    visited_edges = set()

    while True:
        if current not in graph or not graph[current]:
            break
        neighbors = list(graph[current].keys())
        next_node = random.choice(neighbors)
        edge = (current, next_node)

        if edge in visited_edges:
            break
        visited_edges.add(edge)

        path.append(next_node)
        current = next_node

    with open("random_walk.txt", 'w') as f:
        f.write(' '.join(path))
    return path


# 展示有向图（使用networkx仅用于可视化）
def show_directed_graph(graph, save_path=None):
    # 命令行展示
    print("Directed Graph:")
    for node in graph:
        edges = [f"{k}({v})" for k, v in graph[node].items()]
        print(f"  {node} -> {' '.join(edges)}")

    # 创建networkx图对象仅用于绘图
    G = nx.DiGraph()
    for node in graph:
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Directed Graph Visualization")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Graph image saved to {save_path}")

    plt.show(block=False)
    plt.pause(2)


# 主程序
def main():
    file_path = input("Enter text file path: ")
    words = process_text(file_path)
    graph = build_graph(words)

    while True:
        print("\n1. Show graph\n2. Save graph image\n3. Query bridge words\n4. Generate new text")
        print("5. Calculate shortest path\n6. Calculate PageRank\n7. Random walk\n8. Exit")
        choice = input("Choose an option (1-8): ")

        if choice == '1':
            show_directed_graph(graph)
        elif choice == '2':
            save_path = input("Enter save path (without extension): ")
            show_directed_graph(graph, save_path + '.png')
        elif choice == '3':
            w1 = input("Enter first word: ").lower()
            w2 = input("Enter second word: ").lower()
            print(query_bridge_words(graph, w1, w2))
        elif choice == '4':
            text = input("Enter your text: ")
            print("Generated text:", generate_new_text(graph, text))
        elif choice == '5':
            handle_shortest_path_input(graph)
        elif choice == '6':
            pr = calculate_pagerank(graph)
            print("PageRank results (sorted by importance):")
            for word, score in sorted(pr.items(), key=lambda x: -x[1]):
                print(f"{word}: {score:.6f}")
        elif choice == '7':
            walk = random_walk(graph)
            print("Random walk path:", ' '.join(walk))
            print("Path saved to random_walk.txt")
        elif choice == '8':
            plt.close('all')
            print("Exiting program...")
            break
        else:
            print("Invalid option, please try again.")


if __name__ == "__main__":
    try:
        import matplotlib

        matplotlib.use('TkAgg')
    except:
        pass
    main()