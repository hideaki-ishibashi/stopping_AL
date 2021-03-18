import hashlib
import argparse
import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
from stopping_criteria import *
import matplotlib.pylab as plt
import matplotlib.cm as cm
import draw_result
import random

from batch_bald.models.artificial_net import BayesianNet
from batch_bald import multi_bald

parser = argparse.ArgumentParser(description='Bayesian batch active learning as sparse subset approximation')
parser.add_argument('--gpu', default=-1, type=int, metavar='N',
                    help='gpu id')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of epochs for training')
parser.add_argument('--train_batch_size', default=2000, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--test_batch_size', default=128, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--lr', default=0.01, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--scheduling', default='(10,15)', type=str, metavar='N',
                    help='learning rate')
parser.add_argument('--p', default=0.5, type=float, metavar='N',
                    help='Dropout probability of an element to be zeroed')
parser.add_argument('--initial_size', default=1, type=int, metavar='N',
                    help='the size of dataset size before acuquisiton')
parser.add_argument('--acquisition_size', default=5, type=int, metavar='N',
                    help='the size of data points acquisition for each round')
parser.add_argument('--round', default=100, type=int, metavar='N',
                    help='total round for data points acquisition')
parser.add_argument('--seed', default=1, type=int, metavar='N',
                    help='seed value for initial data points')
parser.add_argument('--selection', type=str, default='batchbald',
                    choices=['random', 'batchbald'],
                    help='Types of data points selection. "random" is random selection and "coreset" is the method of sparse subset approximation')
parser.add_argument('--num_features', default=128, type=int, metavar='N',
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
num_class = 2

def get_mixture_of_gaussian(classnum):
    train_size = 20000
    test_size = 1000
    dim = 2
    N = train_size + test_size
    Nc = int(N/classnum)
    Nlast = N - (classnum-1)*Nc
    Xs = []
    ys = []
    r = 4
    for c in range(classnum):
        mean = r*np.random.randn(dim)
        if c != classnum-1:
            Xs.append(np.random.randn(Nc,dim)-mean[None,:])
            ys.append(c*np.ones(Nc))
        else:
            Xs.append(np.random.randn(Nlast,dim)-mean[None,:])
            ys.append(c * np.ones(Nlast))
    X = np.concatenate(Xs,axis=0)
    y = np.concatenate(ys,axis=0).astype("int")
    indecies = random.sample(range(X.shape[0]), X.shape[0])
    X = X[indecies]
    y = y[indecies]
    X = torch.from_numpy(X.astype(np.float32)).clone()
    y = torch.from_numpy(y.astype(np.int)).clone()
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    # train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    return [[X_train, y_train],test_dataset]


def generate_uniform_distribution(sigma=0.1):
    train_size = 20000
    test_size = 1000
    dim = 1
    X = 2*(np.random.rand(train_size+test_size,dim)-0.5)
    z = np.sin(1.8*np.pi*X).prod(axis=1)+np.random.normal(0,sigma,train_size+test_size)
    y = (0.5*(np.sign(z)+1)).astype("int")
    X = torch.from_numpy(X.astype(np.float32)).clone()
    y = torch.from_numpy(y.astype(np.int)).clone()
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    # train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    return [[X_train, y_train],test_dataset]

def generate_full_dataset():
    return datasets.MNIST('../mnist_data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))


def modify_dataset(train_dataset, indices):
    return torch.utils.data.TensorDataset(train_dataset[0][indices],train_dataset[1][indices])

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

    sum_accuracy = sum_accuracy / len(test_loader.dataset)
    sum_mse = sum_mse / len(test_loader.dataset)

    return sum_accuracy, sum_mse

# consistent initial data points based on seed value
[pool_dataset,test_dataset] = generate_uniform_distribution()
total_size = len(pool_dataset[0])
indices = np.random.RandomState(seed=args.seed).permutation(total_size)[:args.initial_size]
print('md5 of initial data points: {}'.format(hashlib.md5(str(indices).encode('utf-8')).hexdigest()))
print(total_size)
train_dataset = modify_dataset(pool_dataset, indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


net = BayesianNet(input_dim=1,hidden_dim=128,num_class=num_class)
net = net.to(device)

print('Selection: {}, Round: {}, Size: {}'.format(args.selection, 0, len(train_loader.dataset)))
KL_list = np.empty(0,float)
test_accs_list = np.empty(0,float)
test_mses_list = np.empty(0,float)

threshold = 0.04
error_stability = error_stability_criterion(threshold)
criteria = [azuma]
criterion = torch.nn.CrossEntropyLoss()
color = {azuma.criterion_name: "r"}
loss_lists = np.zeros([args.round,epochs])
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduling, gamma=0.1)
for r in range(args.round):

    test_mses, test_accs = [], []
    loss_list = np.zeros(epochs)
    for epoch in range(epochs):
        loss = train_one_epoch(net, optimizer, train_loader, criterion, device)
        loss_list[epoch] = loss

    for epoch in range(epochs):
        sum_accracy, sum_mse = test_one_epoch(net, test_loader, device)
        test_accs.append(sum_accracy)
        test_mses.append(sum_mse)
        # scheduler.step()
    loss_lists[r] = loss_list

    [weight_new,bias_new] = net.get_params()
    sigma = 1
    if r == 0:
        weight_old = weight_new.clone()
        bias_old = bias_new.clone()
    if r > 0:
        KL = p / (2 * sigma ** 2) * ((weight_new - weight_old) ** 2).sum() + 1 / (2 * sigma ** 2) * ((bias_new - bias_old) ** 2).sum()
        KL = KL.to('cpu').detach().numpy().copy()
        azuma.check_threshold(KL,KL,r-1)
        print(azuma.R[-1])
        print(azuma.criterion[-1])
        KL_list = np.append(KL_list,KL)
        test_accs_list = np.append(test_accs_list,test_accs[-1])
        test_mses_list = np.append(test_mses_list,test_mses[-1])

    weight_old = weight_new.clone()
    bias_old = bias_new.clone()

    print('Round: {}, Accuracy: {}'.format(r, test_accs[-1]))
    print('Round: {}, MSE: {}'.format(r, test_mses[-1]))
    if args.selection == 'random':
        unlabeled_indices = np.array(list(set(np.arange(len(pool_dataset[0]))).difference(indices)))
        # update indices
        indices = np.concatenate((indices, unlabeled_indices[np.random.permutation(len(unlabeled_indices))][:args.acquisition_size]))
    elif args.selection == 'batchbald':
        # generate new indices
        unlabeled_indices = np.array(list(set(np.arange(len(pool_dataset[0]))).difference(indices)))
        unlabeled_dataset = modify_dataset(pool_dataset, unlabeled_indices)
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
    train_dataset = modify_dataset(pool_dataset, indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    for data in train_loader:
        print(data[0].shape)
    print('Selection: {}, Round: {}, Size: {}'.format(args.selection, r + 1, len(train_loader.dataset)))

print(loss_lists.shape)
for i in range(args.round):
    plt.plot(range(epochs),loss_lists[i],c=cm.rainbow(i/args.round))
plt.xlabel("epochs", fontsize=24)
plt.ylabel("train loss", fontsize=24)
plt.savefig("result/losses.pdf")
plt.pause(0.01)
plt.clf()
plt.plot(range(args.round),loss_lists[:,-1],c="k")
plt.xlabel("data size", fontsize=24)
plt.ylabel("train loss", fontsize=24)
plt.savefig("result/loss.pdf")
plt.pause(0.01)
draw_result.draw_accuracy(test_accs_list, criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_accuracy_MNIST.pdf")
draw_result.draw_mse(test_mses_list, criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_MSE_MNIST.pdf")
draw_result.draw_upper_bound(criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_upper_bound_MNIST.pdf")
draw_result.draw_epsilon(criteria, color,args.acquisition_size)
plt.tight_layout()
plt.savefig("result/BayesianDNN_epsilon_MNIST.pdf")
