import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

#---------------------------------------------------------
def init_model(arch, device, learning_rate, hidden_units=None):
    if arch == 'vgg19':
        if hidden_units == None:
            hidden_units = 4096
        model = models.vgg19(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        model.to(device)
        for param in model.parameters():
            param.require_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch == 'resnet34':
        if hidden_units == None:
            hidden_units = 256
        model = models.resnet34(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.fc = classifier
        model.to(device)
        for param in model.parameters():
            param.require_grad = False
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        print('Error! Model architecture not predefined as an option!')
        
    return model, optimizer
#---------------------------------------------------------
def save_checkpoint(save_dir, model, optimizer, epochs, hidden_units=None):
    model = model.cpu()
    try:
        if hidden_units == None:
            hidden_units = 4096
        checkpoint_cpu = {
            'model_arch': 'vgg19',
            'input_size': 25088, 'output_size': 102, 
            'hidden_layers': hidden_units, # hidden_units
            'state_dict': model.classifier.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optim_state_dict': optimizer.state_dict(),
            'epochs': epochs}
    except:
        if hidden_units == None:
            hidden_units = 256
        checkpoint_cpu = {
            'model_arch': 'resnet34',
            'input_size': 512, 'output_size': 102, 
            'hidden_layers': hidden_units,   # hidden_units
            'state_dict': model.fc.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optim_state_dict': optimizer.state_dict(),
            'epochs': epochs}
        
    torch.save(checkpoint_cpu, save_dir)
    
#---------------------------------------------------------
def load_model(filepath, device, to_train=False):
    
    checkpoint = torch.load(filepath)
    if checkpoint['model_arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif checkpoint['model_arch'] == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        print('Error! Model architecture undefined!')
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    classifier.load_state_dict(checkpoint['state_dict'])
    
    if checkpoint['model_arch'] == 'vgg19':
        model.classifier = classifier
        model.to(device)
        if to_train:
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
    elif checkpoint['model_arch'] == 'resnet34':
        model.fc = classifier
        model.to(device)
        if to_train:
            optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    trained_eps = checkpoint['epochs']
    
    if to_train:
        return model, optimizer, trained_eps
    else:
        return model
#---------------------------------------------------------