import os
import cv2
import numpy as np
from sklearn.metrics import mutual_info_score
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    from torchvision import transforms, datasets
except Exception as e:
    raise ImportError('Please install torch and torchvision.')

batch_size = 32
data_path = '/home/ks01/wangyu/dataset/single_data'
save_path = "/home/ks01/wangyu/anom_detc/Banding-Pattern-Extraction/model_save/resnet.pth"

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101()
        self.features = None

    def forward(self, x):
        return self.resnet101(x)

    def get_features_hook(self, module, input, output):
        self.features = input[0].view(input[0].size(0), -1)


def _gen_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # normalization from ImageNet
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def train(num_epochs=5, learning_rate=0.0003):
    model = ResNet101()
    if os.path.exists(save_path):
        print('Loading pretrained model..')
        model.load_state_dict(torch.load(save_path))
    data_loader = _gen_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def get_feature_from_resnet(input_data):
    """

    :param input_data: 224*224*3
    :return:
    """
    model = ResNet101()
    if not os.path.exists(save_path):
        raise BlockingIOError('Model hasn\'t been trained.')
    model.load_state_dict(torch.load(save_path))
    model.eval()  # set the model to evaluation mode

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_data = transform(input_data).unsqueeze(0)

    hook_handle = model.resnet101.fc.register_forward_hook(model.get_features_hook)
    output = model(input_data)
    input_features = model.features
    hook_handle.remove()

    return np.squeeze(input_features.detach().numpy())


def _cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def calculate_feature_distance(feature1, feature2, method='mutual_info'):
    """

    :param feature1:
    :param feature2:
    :param method: mutual_info or similarity
    :return:
    """
    if method == 'mutual_info':
        return mutual_info_score(feature1, feature2)
    elif method == 'similarity':
        return _cosine_similarity(feature1, feature2)
    else:
        raise ValueError('unsupported method')

