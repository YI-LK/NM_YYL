import sys
import matplotlib.pyplot as plt

def main():
    # 从命令行参数获取输入文件、连接标志和颜色
    if len(sys.argv) < 3:
        print("Usage: python3 plot.py <data_file> <connect> [--color <color1>] [--color <color2>] ...")
        return

    data_file = sys.argv[1]
    connect = int(sys.argv[2])

    # 解析颜色参数
    colors = []
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--color" and i + 1 < len(sys.argv):
            colors.append(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            return

    # 读取数据
    x = []
    y_columns = []  # 存储每一列 y 数据
    with open(data_file, 'r') as f:
        first_line = f.readline().strip().split()
        num_y_columns = len(first_line) - 1  # 计算有多少列 y 数据
        y_columns = [[] for _ in range(num_y_columns)]  # 初始化 y 数据列表

        # 重新处理第一行数据
        x.append(float(first_line[0]))
        for i in range(num_y_columns):
            y_columns[i].append(float(first_line[i + 1]))

        # 继续读取剩余行
        for line in f:
            parts = line.strip().split()
            x.append(float(parts[0]))
            for i in range(num_y_columns):
                y_columns[i].append(float(parts[i + 1]))

    # 如果颜色参数不足，填充默认颜色
    default_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    if len(colors) < num_y_columns:
        colors.extend(default_colors[len(colors):])

    # 设置点的大小（例如 10）
    point_size = 5

    # 绘制散点图和连线
    for i in range(num_y_columns):
        plt.scatter(x, y_columns[i], label=f"y{i+1}", color=colors[i], s=point_size)  # 设置点的大小
        if connect == 0:
            plt.plot(x, y_columns[i], color=colors[i])  # 连线

    plt.legend()  # 显示图例
    plt.show()

if __name__ == "__main__":
    main()