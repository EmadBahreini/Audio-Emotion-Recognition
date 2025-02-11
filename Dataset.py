import pandas as pd
from torch.utils.data import Dataset

class AudioEmotionDataset(Dataset):
    def __init__(self,pick_file:str,transform=None,filter_classes:list=None):
        self.data = pd.read_pickle(pick_file)
        if filter_classes is not None:
            self.data = self.data[self.data.Emotion.isin(filter_classes)]
        self.data = self.data.sample(frac=1,random_state = 42).reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = self.data.Feature[index]
        label = self.data.Emotion[index] # Labels are 1-8, convert to 0-7
        label = label - 1
        if self.transform:
            features = self.transform(features)
        
        return features.unsqueeze(0), label