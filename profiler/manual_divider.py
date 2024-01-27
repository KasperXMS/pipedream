import os
root_path = './profiler/profiles/'

def divide_graph(model_name, split_point):
    graph_file = os.path.join(root_path, model_name, 'graph.txt')
    f1 = open(graph_file, 'r')
    f2 = open(os.path.join(root_path, model_name, 'graph_splitted.txt'), 'w')
    line = f1.readline()
    stage = 0
    while len(line) > 0:
        if line.startswith('node'):
            line = line[:-1]
            node_no = int(line.split(' ')[0][4:])
            line += f' -- stage_id={stage}\n'
            if stage < len(split_point) and node_no == split_point[stage] - 1:
                stage += 1
        
        f2.write(line)
        line = f1.readline()
    
    f1.close()
    f2.close()

if __name__ == '__main__':
    divide_graph('resnet18', [11, 23, 45, 60])