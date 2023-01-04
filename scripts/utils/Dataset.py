import torch

class Dataset:
  
  def __init__(self, url, labels):
    self.urls = url
    self.labels = labels
  
  def __len__(self):
    return len(self.urls)

  def __getitem__(self, index):
    urls = self.urls[index]
    labels = self.labels[index]
    return {'token_id' : torch.IntTensor(urls), 'labels' : torch.tensor(labels, dtype = torch.float)}