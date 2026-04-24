import torch
import torch.nn as nn
from torchvision import transforms

class FER_DCNN(nn.Module):
    def __init__(self):
        super(FER_DCNN, self).__init__()

        self.features = nn.Sequential(
            
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 4 (deeper like paper)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = FER_DCNN()
model.load_state_dict(torch.load("best_fer_model.pth")) ##model path
model.to(device)
model.eval()


from PIL import Image

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open("C:/Users/haris/Downloads/fer2013/test/angry/PrivateTest_10131363.jpg") ##image name to predict
img = transform(img).unsqueeze(0).to(device)


with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(classes[pred.item()])
