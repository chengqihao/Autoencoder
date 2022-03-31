import numpy as np
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import dataset
import loss
import Sinkhorn
import model
import torch

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005  # learning rate
DOWNLOAD_MNIST = True
N_TEST_IMG = 5
eps = 0.1
# Load data
train_data = torchvision.datasets.MNIST(
        root='./dataset/mnist/',
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=False,  # download it if you don't have it
    )

train_loader = dataset.load_train_data(batch_size=BATCH_SIZE, download=DOWNLOAD_MNIST)
test_loader = dataset.load_test_data(batch_size=BATCH_SIZE, download=DOWNLOAD_MNIST)
print(train_loader.dataset.data.size())
print(test_loader.dataset.data.size())

_, _, image_size = train_loader.dataset.data.size()

# Using Single GPU CUDA
is_gpu = torch.cuda.is_available()
gpu_nums = torch.cuda.device_count()
gpu_index = torch.cuda.current_device()
gpu_name = torch.cuda.get_device_name(gpu_index)
print('GPU: ', gpu_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建网络实例
autoencoder = model.AutoEncoder().to(device)
# 设定优化器
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
# Loss Function
loss_func = nn.MSELoss().to(device)
loss_func2 = loss.myLoss().to(device)
# kernel Matrix
M_1D = Sinkhorn.construct_1D_M(image_size=image_size, device=device)
M_2D = Sinkhorn.construct_2D_M(image_size=image_size, device=device)

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()  # continuously plot
view_data = ((train_data.data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor))/255.).to(device)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.cpu().data.numpy()[i], (28, 28)), cmap='gray');
    a[0][i].set_xticks(());
    a[0][i].set_yticks(())


# Training
for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        original_x = x.view(-1, 28 * 28).to(device)  # batch x, shape (batch, 28*28)
        original_y = x.view(-1, 28 * 28).to(device)  # batch y, shape (batch, 28*28)
        bs, _ = original_x.size()
        # 复制核矩阵
        # 1D
        M = M_1D.repeat(bs, 1, 1)
        # 2D
        # M = M_2D.repeat(bs, 1, 1)
        # 神经网络的output
        encoded, decoded = autoencoder(original_x)
        # 计算sinkhorn
        alpha, beta = Sinkhorn.Sinkhorn(original=original_y, target=decoded, M=M, maxiter=50, eps=eps, device=device)
        # 损失函数
        loss = loss_func2(target=decoded, original=original_y, alpha=alpha, beta=beta)  # Sinkhorn
        # 梯度清零
        optimizer.zero_grad()  # clear gradients for this training step
        # 后向传播计算梯度
        loss.backward()  # backpropagation, compute gradients
        # 更新神经网络参数
        optimizer.step()  # apply gradients
        # 每 100批次看一下情况
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())

            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())
            plt.draw();
            plt.pause(0.05)
torch.cuda.empty_cache()
