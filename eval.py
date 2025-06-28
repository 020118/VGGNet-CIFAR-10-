import torch
import argparse
import vgg
from utils import load_data


def eval(trained_model):
    _, test_loader, classes = load_data()

    #展示部分测试集数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    print('GroundTrue:', ' '.join('%5s' % classes[labels[i]] for i in range(4)))

    outputs = trained_model(images)
    _, predicted = torch.max(outputs, dim=1)
    print('Predicted:', ' '.join('%5s' % classes[predicted[i]] for i in range(4)))

    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        images, labels = data
        outputs = trained_model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (labels == predicted).sum()
    
    print('Accuracy of the test dataset: %.2f %%' % (100.0*correct/total))

    #看各个种类的错误和正确数
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in test_loader:
        images, labels = data
        outputs = trained_model(images)
        _, predicted = torch.max(outputs, dim=1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s %.2f %%' % (classes[i], 100*class_correct[i]/class_total[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=False, default='vgg19.pth')
    
    args = parser.parse_args()
    print('started model:', args.checkpoint_path[:-4])

    trained_net = vgg.__dict__[args.checkpoint_path[:-4]]()
    trained_net.load_state_dict(torch.load(args.checkpoint_path))
    
    eval(trained_net)
