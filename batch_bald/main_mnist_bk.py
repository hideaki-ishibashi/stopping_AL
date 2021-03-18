import hashlib
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from batch_bald.models.mnist_model import BayesianNet
from stopping_criteria import azuma_criterion
import matplotlib.pylab as plt
import draw_result
from batch_bald import multi_bald

parser = argparse.ArgumentParser(description='Bayesian batch active learning as sparse subset approximation')
parser.add_argument('--gpu', default=-1, type=int, metavar='N',
                    help='gpu id')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of epochs for training')
parser.add_argument('--train_batch_size', default=10000, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--test_batch_size', default=128, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--scheduling', default='(10,15)', type=str, metavar='N',
                    help='learning rate')
parser.add_argument('--p', default=0.5, type=float, metavar='N',
                    help='Dropout probability of an element to be zeroed')
parser.add_argument('--initial_size', default=1000, type=int, metavar='N',
                    help='the size of dataset size before acuquisiton')
parser.add_argument('--test_size', default=10000, type=int, metavar='N',
                    help='the size of test dataset')
parser.add_argument('--acquisition_size', default=3, type=int, metavar='N',
                    help='the size of data points acquisition for each round')
parser.add_argument('--round', default=100, type=int, metavar='N',
                    help='total round for data points acquisition')
parser.add_argument('--seed', default=1, type=int, metavar='N',
                    help='seed value for initial data points')
parser.add_argument('--selection', type=str, default='batchbald',
                    choices=['random', 'batchbald'],
                    help='Types of data points selection. "random" is random selection and "coreset" is the method of sparse subset approximation')
parser.add_argument('--num_features', default=32, type=int, metavar='N',
                    help='number of features of projections')
parser.add_argument("--num_inference_samples", type=int, default=5, help="number of samples for inference")

args = parser.parse_args()
if args.gpu < 0:
    args.gpu = 'cpu'
device = torch.device(args.gpu)
epochs = args.epochs
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
lr = args.lr
p = args.p

scheduling = eval(args.scheduling)
num_class = 10


def generate_full_dataset():
    return datasets.MNIST('./mnist_data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))


def modify_dataset(train_dataset, indices):
    train_dataset.data = train_dataset.data[indices]
    train_dataset.targets = train_dataset.targets[indices]
    return train_dataset


# consistent initial data points based on seed value
train_dataset = generate_full_dataset()
print(type(train_dataset))
total_size = len(train_dataset)
indices = np.random.RandomState(seed=args.seed).permutation(total_size)[:args.initial_size]
print('md5 of initial data points: {}'.format(hashlib.md5(str(indices).encode('utf-8')).hexdigest()))
train_dataset = modify_dataset(train_dataset, indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

test_dataset = datasets.MNIST('./mnist_data', train=False,transform=transforms.Compose([transforms.ToTensor(),]))
test_indices = np.random.RandomState(seed=args.seed).permutation(len(test_dataset))[:args.test_size]
test_dataset = modify_dataset(test_dataset, test_indices)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


def train_one_epoch(net, optimizer, train_loader,criterion,device, num_samples=args.num_inference_samples):

    net.train()

    loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        out = net(x, k=num_samples)
        out = out.mean(dim=1)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def test_one_epoch(net, test_loader, device, num_samples=args.num_inference_samples):

    net.eval()

    sum_accuracy = 0.0
    sum_mse = 0.0
    sum_loss = 0.0
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = F.softmax(net(x, k=num_samples),dim=-1)
            logits = logits.mean(dim=1)
        # MSE
        y_vector = torch.zeros(logits.shape)
        y_vector[torch.arange(y.shape[0]), y] = 1.0
        y_vector = y_vector.to(logits.device)
        sum_mse += torch.pow(logits - y_vector, 2).sum().item()
        # ACUURACY
        sum_accuracy += (logits.argmax(dim=-1) == y).sum().item()
        # loss
        sum_loss += criterion(logits, y)

    sum_accuracy = sum_accuracy / len(test_loader.dataset)
    sum_mse = sum_mse / len(test_loader.dataset)
    sum_loss = sum_loss / len(test_loader.dataset)

    return sum_accuracy, sum_mse, sum_loss


net = BayesianNet(num_classes=num_class)
net = net.to(device)

print('Selection: {}, Round: {}, Size: {}'.format(args.selection, 0, len(train_loader.dataset)))
KL_list = np.empty(0,float)
test_accs_list = np.empty(0,float)
test_mses_list = np.empty(0,float)
test_loss_list = np.empty(0,float)

threshold = 0.25
azuma1 = azuma_criterion(threshold)
threshold = 0.2
azuma2 = azuma_criterion(threshold)
threshold = 0.15
azuma3 = azuma_criterion(threshold)
criteria = [azuma1, azuma2, azuma3]
criterion = torch.nn.CrossEntropyLoss()
color = {azuma1.criterion_name: "r", azuma2.criterion_name: "g", azuma3.criterion_name: "b"}
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.00)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduling, gamma=0.1)
for r in range(args.round):

    test_mses, test_accs, test_loss = [], [], []

    for epoch in range(epochs):
        train_one_epoch(net, optimizer, train_loader, criterion, device=device)
        sum_accracy, sum_mse, sum_loss = test_one_epoch(net, test_loader, device)
        test_accs.append(sum_accracy)
        test_mses.append(sum_mse)
        test_loss.append(sum_loss)
        # scheduler.step()

    p=0.5
    [weight_new,bias_new] = net.get_params()
    sigma = 1
    if r == 0:
        weight_old = weight_new.clone()
        bias_old = bias_new.clone()
    if r > 0:
        KL = p / (2 * sigma ** 2) * ((weight_new - weight_old) ** 2).sum() + 1 / (2 * sigma ** 2) * ((bias_new - bias_old) ** 2).sum()
        KL = KL.to('cpu').detach().numpy().copy()
        azuma1.check_threshold(KL,KL,r-1)
        azuma2.check_threshold(KL,KL,r-1)
        azuma3.check_threshold(KL,KL,r-1)
        print(azuma1.R[-1])
        print(azuma1.epsilon[-1])
        KL_list = np.append(KL_list,KL)
        test_accs_list = np.append(test_accs_list,test_accs[-1])
        test_mses_list = np.append(test_mses_list,test_mses[-1])
        test_loss_list = np.append(test_loss_list,test_loss[-1])

    weight_old = weight_new.clone()
    bias_old = bias_new.clone()

    print('Round: {}, Accuracy: {}'.format(r, test_accs[-1]))
    print('Round: {}, MSE: {}'.format(r, test_mses[-1]))
    if args.selection == 'random':
        train_dataset = generate_full_dataset()
        unlabeled_indices = np.array(list(set(np.arange(len(train_dataset))).difference(indices)))
        # update indices
        indices = np.concatenate((indices, unlabeled_indices[np.random.permutation(len(unlabeled_indices))][:args.acquisition_size]))
    elif args.selection == 'batchbald':
        # generate new indices
        train_dataset = generate_full_dataset()
        unlabeled_indices = np.array(list(set(np.arange(len(train_dataset))).difference(indices)))
        unlabeled_dataset = modify_dataset(train_dataset, unlabeled_indices)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=train_batch_size, shuffle=False)
        new_indices, _ = multi_bald.compute_multi_bald_batch(
                bayesian_model=net,
                unlabeled_loader=unlabeled_loader,
                num_classes=num_class,
                k=args.num_inference_samples,
                b=args.acquisition_size,
                )
        new_indices = unlabeled_indices[new_indices]
        indices = np.concatenate((indices, new_indices))
    # generate new loader based on updated indices
    print(indices.shape)
    train_dataset = generate_full_dataset()
    train_dataset = modify_dataset(train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    print('Selection: {}, Round: {}, Size: {}'.format(args.selection, r + 1, len(train_loader.dataset)))

draw_result.draw_accuracy(test_accs_list, criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_accuracy_MNIST.pdf")
draw_result.draw_mse(test_mses_list, criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_MSE_MNIST.pdf")
draw_result.draw_loss(test_loss_list, criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_LOSS_MNIST.pdf")
draw_result.draw_upper_bound(criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_upper_bound_MNIST.pdf")
draw_result.draw_epsilon(criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_epsilon_MNIST.pdf")

np.savetxt("result_data/accuracy.txt",np.array(test_accs_list))
np.savetxt("result_data/mse.txt",np.array(test_mses_list))
np.savetxt("result_data/loss.txt",np.array(test_loss_list))
np.savetxt("result_data/lambda.txt",azuma1.epsilon)
np.savetxt("result_data/R_upper.txt",azuma1.R_upper)
np.savetxt("result_data/R_lower.txt",azuma1.R_lower)
if azuma1.stop_timings != np.nan:
    np.savetxt("result_data/stop_timings_azuma1.txt",np.array([azuma1.stop_timings]))
if azuma2.stop_timings != np.nan:
    np.savetxt("result_data/stop_timings_azuma2.txt",np.array([azuma2.stop_timings]))
if azuma3.stop_timings != np.nan:
    np.savetxt("result_data/stop_timings_azuma3.txt",np.array([azuma3.stop_timings]))
