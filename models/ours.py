import torch
import torch.nn as nn

class ScoresModel(nn.Module):

    def __init__(self, in_time: int, num_scores: int, num_classes: int):

        super(ScoresModel, self).__init__()

        assert in_time%num_scores == 0, "in_time and n_scores must be divisible"
        
        stepsize = int(in_time/num_scores)

        self.num_scores = num_scores
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        
        self.conv5 = nn.Conv3d(512, 1024, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)

        self.fc = nn.Linear(1024, num_classes)

        self.scores_conv1 = nn.Conv3d(512, 128, (2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        self.bn_scores1 = nn.BatchNorm3d(128)

        self.scores_conv2 = nn.Conv3d(128, 32, 1, stride=1, padding=0)
        self.bn_scores2 = nn.BatchNorm3d(32)

        self.relu = nn.ReLU()

        self.scores_adptive_avgpool3d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.cls_adptive_avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        batch_size = x.size(0)

        hs = self.backbone(x)
        
        cls_h = self.relu(self.bn5(self.conv5(hs)))
        cls_h = self.cls_adptive_avgpool3d(cls_h)
        cls_h = cls_h.view(batch_size, -1)
        cls_out = self.fc(cls_h)

        scores_h = self.relu(self.bn_scores1(self.scores_conv1(hs)))
        scores_h = self.bn_scores2(self.scores_conv2(scores_h))
        scores_out = self.scores_adptive_avgpool3d(scores_h)

        return cls_out, scores_out

