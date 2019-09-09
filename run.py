from pathlib import Path

import numpy as np
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from dataset import get_dataset, get_handler
from model import get_net
import config
from torchvision import transforms
import torch
import os
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
    AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

# parameters
SEED = 1

NUM_INIT_LB = 10000
NUM_QUERY = 1000
NUM_ROUND = 10

configuration = config.load_config()
net = get_net(configuration.net)()
if configuration.load_model:
    try:
        net.load_state_dict(torch.load(configuration.model_file))
    except:
        print("No model file found at", configuration.model_file)


# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.backends.cudnn.enabled = False

# load dataset
dataset = datasets.ImageFolder(root=configuration.labeled_training_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(configuration.image_size)
                                   , transforms.CenterCrop(configuration.image_size)
                                   , transforms.ToTensor()
                                   , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
classes = dataset.classes
for class_name in classes:
    if not os.path.isdir(Path(configuration.predicted_dir, class_name)):
        os.makedirs(Path(configuration.predicted_dir,class_name), exist_ok=False)
# create dataloader
device = torch.device("cuda:0" if (torch.cuda.is_available() and configuration.num_gpus > 0) else "cpu")

dataloader = DataLoader(dataset, batch_size=configuration.batch_size, shuffle=True,
                        num_workers=configuration.num_workers)

# X_tr, Y_tr, X_te, Y_te = dataset.get_split()

optimizer = optim.SGD(net.parameters(), **configuration.optimizer_args)
net = net.to(device)

# train
losses = []
for epoch in range(configuration.epochs):
    net.train()
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # print('x', x.shape)
        out, e1 = net(x)
        # print("out", out.shape, out)
        # print("y", y.shape, y)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        print(loss)
print(losses[-1])

if configuration.save_model:
    torch.save(net.state_dict(), configuration.model_file)
    print("Model saved")
# get entropy of unlabeled images
'''
unlabeled_dataset = datasets.ImageFolder(root=configuration.unlabeled_dir,
                                         transform=transforms.Compose([
                                             transforms.Resize(configuration.image_size)
                                             , transforms.CenterCrop(configuration.image_size)
                                             , transforms.ToTensor()
                                             , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))

unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False)
'''
transform = transforms.Compose([
    transforms.Resize(configuration.image_size)
    , transforms.CenterCrop(configuration.image_size)
    , transforms.ToTensor()
    , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

unlabeled_images = [file_name for file_name in os.listdir(configuration.unlabeled_dir) if file_name.endswith('.jpg') or file_name.endswith('.png')]
print("preparing", str(len(unlabeled_images)), "unlabeled_images for entropy")
probs = torch.zeros([len(unlabeled_images), 2])  # todo 2 -> num classes
with torch.no_grad():
    for idx, image_name in enumerate(unlabeled_images):
        if idx % 100 == 0:
            print("finding entropy of images", str(idx), " to", str(idx + 100))
        image_path = Path(configuration.unlabeled_dir, image_name)
        image = config.PIL.Image.open(image_path).convert("RGB")
        transformed_image = transform(image)
        transformed_image = transformed_image.unsqueeze(0).to(device)
        # for idx, (x, y) in enumerate(unlabeled_loader):
        # x, y = x.to(device), y.to(device)
        out, e1 = net(transformed_image)
        prob = F.softmax(out, dim=1)
        probs[idx] += prob.cpu()[0]
    probs /= 2

log_probs = torch.log(probs)
U = (probs * log_probs).sum(1)
requests = U.sort()[1]

for request_idx in requests:
    print((unlabeled_images[request_idx]), U[request_idx])

# request human labeling of most confusing unlabeled examples
if configuration.num_to_request > 0:
    for request_idx in requests[:configuration.num_to_request]:
        image_name = unlabeled_images[request_idx]
        old_image_path = Path(configuration.unlabeled_dir, image_name)
        new_image_path = Path(configuration.requested_dir, image_name)
        os.rename(old_image_path, new_image_path)

# predict least confusing unlabeled examples
if configuration.num_to_predict > 0:
    for request_idx in requests[-configuration.num_to_predict:]:
        image_name = unlabeled_images[request_idx]
        image_path = Path(configuration.unlabeled_dir, image_name)
        image = config.PIL.Image.open(image_path).convert("RGB")
        transformed_image = transform(image)
        transformed_image = transformed_image.unsqueeze(0).to(device)
        # for idx, (x, y) in enumerate(unlabeled_loader):
        # x, y = x.to(device), y.to(device)
        out, e1 = net(transformed_image)
        class_name = classes[out.argmax()]

        old_image_path = Path(configuration.unlabeled_dir, image_name)
        new_image_path = Path(configuration.predicted_dir, class_name, image_name)
        os.rename(old_image_path, new_image_path)

print(losses[-1])
