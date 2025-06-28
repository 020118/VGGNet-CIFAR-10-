import torch
import torch.nn as nn
import vgg
import argparse
from utils import *
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)         # 应该是非 None，例如 '12.1'
    print(torch.cuda.is_available())  # 应该是 True 才行

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的：{device}")
    #选出各个vgg模型，如vgg11，vgg13
    model_name = sorted(name for name in vgg.__dict__ if name.startswith("vgg") and not name.endswith("net")) 

    #添加命令行参数的处理
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH',  default='vgg19', choices=model_name, 
                        help='a model for training, default:vgg19')
    args = parser.parse_args()

    #加载数据集
    train_loader, _, _ = load_data()

    #确定模型
    net = vgg.__dict__[args.arch]()
    net = net.to(device)
    print(net)

    #创建日志文件
    f = open(args.arch+'log.txt', mode='wt', encoding='utf-8')
    f.write(args.arch + '\n')
    writer = SummaryWriter(comment=args.arch)

    criterion, optimizer = Optimizer(net)

    for epoch in range(100):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            def compute_acc(output, label):
                prediction = torch.softmax(output, dim=1)
                return (prediction.argmax(dim=1)==label).type(torch.float).mean()

            acc = compute_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

            if i%23 == 22:
                print("[%d %5d] loss: %.3f accuracy: %.3f" % (epoch+1, i/23+1, running_loss/23, running_acc/23))
                f.write("[%d %5d] loss: %.3f accuracy: %.3f" % (epoch+1, i/23+1, running_loss/23, running_acc/23))
                writer.add_scalar('loss/train', running_loss/23, epoch*len(train_loader)+i)
                writer.add_scalar('accuracy/train', running_acc/23, epoch*len(train_loader)+i)

                running_acc = 0.0
                running_loss = 0.0

    writer.close()
    f.close()
    print("Finished Training")
    PATH = './' + args.arch + '.pth'
    torch.save(net.state_dict(), PATH)

