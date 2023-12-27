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
from main import profile_train
import os

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
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if i >= 0:
            break
    graph_creator.unhook_modules()

    # persistance of the model graph i.e. record the info into file
    graph_creator.persist_graph(directory)

def main():

    # build model -- AlexNet
    model = AlexNet()
    # Since AlexNet has a paralleled structure, the features should be processed for parallel training
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    train_dataset = SyntheticDataset((3, 224, 224), train_dataset_len)

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
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

    