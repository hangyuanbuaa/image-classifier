from neural_networks import init_model, load_model, save_checkpoint
import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm

#---------------------------------------------------------
def acc_eval(log_ps, labels):

    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    acc = torch.mean(equals.type(torch.FloatTensor))

    return acc

#---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, default='flowers')
    parser.add_argument('--last_checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--save_dir', nargs=1, default='checkpoint.pth')
    parser.add_argument('--arch', type=str, default='vgg19')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--retrain', action='store_true')

    args = parser.parse_args()

    data_dir = ''.join(args.data_dir)
    if args.retrain:
        last_checkpoint = ''.join(args.last_checkpoint)
    save_dir = ''.join(args.save_dir)
    arch = ''.join(args.arch)
    epochs = int(args.epochs)
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'train': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train']),
        'valid': datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid']),
        'test':  datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"

    if args.retrain:
        model, optimizer, trained_eps = load_model(last_checkpoint, device, to_train=True)
        print('Load {} and rebuild model previously trained with () epochs!'.format(args.last_checkpoint, trained_eps))
    else:
        model, optimizer = init_model(arch, device, learning_rate, hidden_units)
        trained_eps = 0
        print('Model initiated with architecture {}!'.format(args.arch))

    criterion = nn.NLLLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    train_losses, valid_losses = [], []
    for e in range(epochs):
        train_loss, train_acc = 0, 0
        scheduler.step()
        print('Epoch: {}/{}\n'.format(e+1+trained_eps, epochs+trained_eps),
              'Training...')
        for images, labels in tqdm(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc_eval(log_ps, labels)

        avg_train_loss = train_loss/len(dataloaders['train'])
        train_losses.append(avg_train_loss)

        print('train_loss: {:.3f}  '.format(avg_train_loss),
              'train_acc: {:.2f} % \n\n'.format(100*train_acc/len(dataloaders['train'])),
              'Next: validation... ')

        valid_loss, valid_acc = 0, 0
        with torch.no_grad():
            model.eval()
            for images, labels in tqdm(dataloaders['valid']):
                images, labels = images.to(device), labels.to(device)
                log_ps = model.forward(images)
                valid_loss += criterion(log_ps, labels).item()
                valid_acc += acc_eval(log_ps, labels)
            model.train()
            avg_valid_loss = valid_loss/len(dataloaders['valid'])
            valid_losses.append(avg_valid_loss)
            print('valid_loss: {:.3f}  '.format(avg_valid_loss),
                  'valid_acc: {:.2f} % \n'.format(100*valid_acc/len(dataloaders['valid'])))
    else:
        model.class_to_idx = image_datasets['train'].class_to_idx
        try:
            save_checkpoint(save_dir, model, optimizer, epochs+trained_eps, hidden_units)
            print('Model training completed! Checkpoint saved in {}!'.format(args.save_dir))
        except:
            print('Model training completed! No checkpoint saved!')
