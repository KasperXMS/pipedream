import sys; sys.path = [".."] + sys.path
from torchvision.models import AlexNet
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from collections import OrderedDict
import torchmodules.torchsummary as torchsummary
import torchmodules.torchgraph as torchgraph
import torchmodules.torchprofiler as torchprofiler
import os, time

# some hyperparameters
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
num_workers = 4
batch_size = 256
train_dataset_len = 1000

# SyntheticDataset in main.py
# generate a random set of input data
class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length

# Migrated from main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# migrated from main.py
# create the model's graph based on summary
def create_graph(model, train_loader, summary, directory):
    """Given a model, creates and visualizes the computation DAG
       of the model in the passed-in directory."""
    graph_creator = torchgraph.GraphCreator(model, summary, module_whitelist=[])
    # wrapper function inside, its role is as following:
    # Wrapper function to "forward()", keeping track of dependencies.
    graph_creator.hook_modules(model)
    for i, (input, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = model(input)
        if i >= 0:
            break
    graph_creator.unhook_modules()

    # persistance of the model graph i.e. record the info into file
    graph_creator.persist_graph(directory)

# migrated from main.py
# run a train process to calculate the forwarding time
def profile_train(train_loader, model, criterion, optimizer):
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    NUM_STEPS_TO_PROFILE = 100  # profile 100 steps or minibatches

    # switch to train mode
    model.train()

    layer_timestamps = []
    data_times = []

    iteration_timestamps = []
    opt_step_timestamps = []
    data_timestamps = []
    
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_pid = os.getpid()
        data_time = time.time() - start_time
        data_time_meter.update(data_time)
        with torchprofiler.Profiling(model, module_whitelist=[]) as p:
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            if isinstance(output, tuple):
                loss = sum((criterion(output_elem, target) for output_elem in output))
            else:
                loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer_step_start = time.time()
            optimizer.step()

            end_time = time.time()
            iteration_time = end_time - start_time
            batch_time_meter.update(iteration_time)

            if i >= NUM_STEPS_TO_PROFILE:
                break
        p_str = str(p)
        layer_timestamps.append(p.processed_times())
        data_times.append(data_time)

        print('End-to-end time: {batch_time.val:.3f} s ({batch_time.avg:.3f} s)'.format(
                  batch_time=batch_time_meter))

        iteration_timestamps.append({"start": start_time * 1000 * 1000,
                                     "duration": iteration_time * 1000 * 1000})
        opt_step_timestamps.append({"start": optimizer_step_start * 1000 * 1000,
                                    "duration": (end_time - optimizer_step_start) * 1000 * 1000, "pid": os.getpid()})
        data_timestamps.append({"start":  start_time * 1000 * 1000,
                                "duration": data_time * 1000 * 1000, "pid": data_pid})
        
        start_time = time.time()

    layer_times = []
    tot_accounted_time = 0.0
    print("\n==========================================================")
    print("Layer Type    Forward Time (ms)    Backward Time (ms)")
    print("==========================================================")

    for i in range(len(layer_timestamps[0])):
        layer_type = str(layer_timestamps[0][i][0])
        layer_forward_time_sum = 0.0
        layer_backward_time_sum = 0.0
        for j in range(len(layer_timestamps)):
            layer_forward_time_sum += (layer_timestamps[j][i][2] / 1000)
            layer_backward_time_sum += (layer_timestamps[j][i][5] / 1000)
        layer_times.append((layer_type, layer_forward_time_sum / len(layer_timestamps),
                                    layer_backward_time_sum / len(layer_timestamps)))
        print(layer_times[-1][0], layer_times[-1][1], layer_times[-1][2])
        tot_accounted_time += (layer_times[-1][1] + layer_times[-1][2])

    print()
    print("Total accounted time: %.3f ms" % tot_accounted_time)
    return layer_times, (sum(data_times) * 1000.0) / len(data_times)


def main():

    # build model -- AlexNet
    model = AlexNet()
    # Since AlexNet has a paralleled structure, the features should be processed for parallel training
    model.features = torch.nn.DataParallel(model.features)
    if torch.cuda.is_available():
        model.cuda()

    train_dataset = SyntheticDataset((3, 224, 224), train_dataset_len)

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    
    # Setup train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    
    # collect model input
    print("Collecting profile...")
    for i, (model_input, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            model_input = model_input.cuda()
        if i >= 0:
            break
    
    print("Doing summarization...")
    # do model summary for obtain each layer's parameter info
    summary = torchsummary.summary(model=model, module_whitelist=[], model_input=(model_input,),
                                    verbose=True, device="cuda")
    print("Summarization done!")
    print("Obtaining forwarding and backforwarding time...")
    # do a set of training to obtain timing info
    per_layer_times, data_time = profile_train(train_loader, model, criterion, optimizer)
    print("time obtained!")
    summary_i = 0
    per_layer_times_i = 0
    while summary_i < len(summary) and per_layer_times_i < len(per_layer_times):
        summary_elem = summary[summary_i]
        per_layer_time = per_layer_times[per_layer_times_i]
        if str(summary_elem['layer_name']) != str(per_layer_time[0]):
            summary_elem['forward_time'] = 0.0
            summary_elem['backward_time'] = 0.0
            summary_i += 1
            continue
        summary_elem['forward_time'] = per_layer_time[1]
        summary_elem['backward_time'] = per_layer_time[2]
        summary_i += 1
        per_layer_times_i += 1
    summary.append(OrderedDict())
    summary[-1]['layer_name'] = 'Input'
    summary[-1]['forward_time'] = data_time
    summary[-1]['backward_time'] = 0.0
    summary[-1]['nb_params'] = 0.0
    summary[-1]['output_shape'] = [batch_size] + list(model_input.size()[1:])
    print("Creating graph...")

    # create grapth and persist it into file
    create_graph(model, train_loader, summary,
                    os.path.join("profiles/", 'alexnet'))
    print("...done!")

if __name__ == '__main__':
    main()

    