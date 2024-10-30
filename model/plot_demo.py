import networkx as nx
import matplotlib.pyplot as plt


def create_standard_graph(row_count, col_count):
    """创建一个标准图结构，分成多排，每排的节点两两相连，并与对面节点相连"""
    G = nx.Graph()

    # 添加节点
    for row in range(row_count):
        for col in range(col_count):
            G.add_node((row, col))

    # 添加每排的节点之间的边
    for row in range(row_count):
        for col in range(col_count - 1):
            G.add_edge((row, col), (row, col + 1))  # 连接同一排的节点

    # 添加对面节点之间的边
    for row in range(row_count - 1):
        for col in range(col_count):
            G.add_edge((row, col), (row + 1, col))  # 连接上下排同一列的节点
            if col > 0:
                G.add_edge((row, col), (row + 1, col - 1))  # 连接左侧节点
            if col < col_count - 1:
                G.add_edge((row, col), (row + 1, col + 1))  # 连接右侧节点

    return G


def draw_graph(G):
    """绘制给定的 NetworkX 图"""
    plt.figure(figsize=(20, 6))
    pos = {(row, col): (col, -row) for row, col in G.nodes()}  # 设置节点位置
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=6, font_color='black')
    plt.title("Standard Graph Structure")
    plt.show()


# 示例用法
if __name__ == "__main__":
    row_count = 2  # 排数
    col_count = 20  # 每排的节点数
    G = create_standard_graph(row_count, col_count)
    draw_graph(G)
