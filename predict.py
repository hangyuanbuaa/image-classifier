import numpy as np
import pandas as pd
import seaborn as sb
from PIL import Image
import torch
import matplotlib
import matplotlib.pyplot as plt
from neural_networks import load_model
from utils import process_image, imshow
import json
import argparse
import time

#---------------------------------------------------------
def predict(image, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0)
    
    image, model = image.to(device), model.to(device)

    model.eval()
    log_ps = model.forward(image)
    ps = torch.exp(log_ps)
    
    probs, classes = ps.topk(topk, dim=1)
    return probs, classes
#---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('checkpoint', type=str, default='vgg19.pth')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', nargs='?', default='cat_to_name.json')
    args = parser.parse_args() 
    
    image_path = ''.join(args.image_path)
    checkpoint = ''.join(args.checkpoint)
    file = ''.join(args.category_names)
    top_k = int(args.top_k)

    with open(file, 'r') as f:  
        cat_to_name = json.load(f)

    image = Image.open(image_path)
    start = time.time()

    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"
    # load checkpoint and rebuild model
    model = load_model(args.checkpoint, device)    
    # make prediction and map the category to flower species
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    probs, labels = predict(image, model, device, topk=top_k)
    
    probs, labels = probs.cpu(), labels.cpu()
    probs, labels = probs.detach().numpy().tolist()[0], labels.detach().numpy().tolist()[0]
    classes = [idx_to_class[idx] for idx in labels]

    data = pd.DataFrame({'probability': probs, 'class': classes, 'label': labels})
    data['flower'] = data['class'].map(cat_to_name)
    print(data[['flower','probability','class']])
    print('Time used: {:.2f} s'.format(time.time()-start))

if __name__ == "__main__":
    main()
