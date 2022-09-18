import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_trnsform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
trainset_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_trnsform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_trnsform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("cifar10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()