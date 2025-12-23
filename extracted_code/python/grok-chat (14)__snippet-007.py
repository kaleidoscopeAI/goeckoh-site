def __init__(self):
    super().__init__()
    # Improved CREPE CNN (torch, CPU; from paper: 6 conv, BN, dropout, maxpool; adjusted kernels to avoid size0)
    self.conv1 = nn.Conv1d(1, 1024, kernel_size=512, stride=4)
    self.bn1 = nn.BatchNorm1d(1024)
    self.dropout1 = nn.Dropout(0.25)
    self.pool1 = nn.MaxPool1d(kernel_size=2)

    self.conv2 = nn.Conv1d(1024, 128, kernel_size=64, stride=1)
    self.bn2 = nn.BatchNorm1d(128)
    self.dropout2 = nn.Dropout(0.25)
    self.pool2 = nn.MaxPool1d(kernel_size=2)

    self.conv3 = nn.Conv1d(128, 128, kernel_size=64, stride=1)
    self.bn3 = nn.BatchNorm1d(128)
    self.dropout3 = nn.Dropout(0.25)
    self.pool3 = nn.MaxPool1d(kernel_size=2)

    self.conv4 = nn.Conv1d(128, 256, kernel_size=64, stride=1)
    self.bn4 = nn.BatchNorm1d(256)
    self.dropout4 = nn.Dropout(0.25)
    self.pool4 = nn.MaxPool1d(kernel_size=2)

    self.conv5 = nn.Conv1d(256, 512, kernel_size=64, stride=1)
    self.bn5 = nn.BatchNorm1d(512)
    self.dropout5 = nn.Dropout(0.25)
    self.pool5 = nn.MaxPool1d(kernel_size=2)

    self.conv6 = nn.Conv1d(512, 1024, kernel_size=64, stride=1)
    self.bn6 = nn.BatchNorm1d(1024)
    self.dropout6 = nn.Dropout(0.25)
    self.pool6 = nn.MaxPool1d(kernel_size=2)

    self.fc = nn.Linear(1024, 360)
    self.sigmoid = nn.Sigmoid()

def forward(self, x):
    x = self.pool1(self.dropout1(self.bn1(torch.relu(self.conv1(x)))))
    x = self.pool2(self.dropout2(self.bn2(torch.relu(self.conv2(x)))))
    x = self.pool3(self.dropout3(self.bn3(torch.relu(self.conv3(x)))))
    x = self.pool4(self.dropout4(self.bn4(torch.relu(self.conv4(x)))))
    x = self.pool5(self.dropout5(self.bn5(torch.relu(self.conv5(x)))))
    x = self.pool6(self.dropout6(self.bn6(torch.relu(self.conv6(x)))))
    x = x.mean(dim=2)  # Global avg
    x = self.sigmoid(self.fc(x))
    return x

