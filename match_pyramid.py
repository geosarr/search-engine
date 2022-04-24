import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from termcolor import colored
from torch.utils.data import Dataset
import torch.optim as optim
from torchtext.vocab import GloVe, vocab
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class MsmarcoDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    item = {
        "query": np.array(self.data['query'][idx]),
        "doc": np.array(self.data['doc'][idx]),
        "label": np.array(self.data['label'][idx], dtype='float')
    }
    return item


class MatchPyramid(torch.nn.Module):
    def __init__(self, num_conv2d=1, pool_size=(2,2), filters=1, kernel_size=5, padding="same", hidden_dim=32, pretrained_vectors=None, pad_index=1):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MatchPyramid, self).__init__()
        self.num_conv2d = num_conv2d
        self.hidden_dim = hidden_dim
        self.ebd = torch.nn.Embedding.from_pretrained(pretrained_vectors, freeze=True, padding_idx=pad_index)
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=kernel_size, padding=padding)
        self.pool2d = torch.nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(p=0.2)
        self.hidden = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, d):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # apply the pretrained embeddings
        q  = self.ebd(q)
        d  = self.ebd(d)

        # compute matrix interaction between query and document
        q = q/q.norm(dim=1)[:, None]
        d = d/d.norm(dim=1)[:, None]
        M = torch.mm(q[0],d[0].transpose(0,1))

        # apply consecutive conv2d and pool2d
        z = self.conv2d(M[None,None,:])
        z = torch.relu(z)
        z = self.pool2d(z)

        for i in range(self.num_conv2d):
          z = self.conv2d(z)
          z = torch.relu(z)
          z = self.pool2d(z)
        
        # build multilayer perceptron with dropout
        pool_flat = torch.flatten(z)
        pool_flat = self.dropout(pool_flat)
        mlp = torch.nn.Linear(pool_flat.shape[0], self.hidden_dim, bias=True)(pool_flat)
        mlp= torch.relu(mlp)
        out = self.out(mlp)

        logit = self.sigmoid(out)

        return logit


msmarco=load_dataset('ms_marco', 'v1.1')

pretrained_vectors = GloVe(name="6B", dim='50')
pretrained_vocab = vocab(pretrained_vectors.stoi)
unk_token = "<unk>"
unk_index = 0
pad_token = '<pad>'
pad_index = 1
pretrained_vocab.insert_token("<unk>",unk_index)
pretrained_vocab.insert_token("<pad>", pad_index)
# this is necessary otherwise it will throw runtime error if OOV token is queried 
pretrained_vocab.set_default_index(unk_index)
pretrained_embeddings = pretrained_vectors.vectors
pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
pretrained_embeddings.size()


tok = TweetTokenizer()
def tokenize_pad_numericalize(entry, vocab_stoi, max_length=20):
  text = [ vocab_stoi[token] if token in vocab_stoi else vocab_stoi['<unk>'] for token in tok.tokenize(entry.lower())]
  padded_text = None
  if len(text) < max_length:   padded_text = text + [ vocab_stoi['<pad>'] for i in range(len(text), max_length) ] 
  elif len(text) > max_length: padded_text = text[:max_length]
  else:                        padded_text = text
  return padded_text

def tokenize_all(entries, vocab_stoi):
  return [tokenize_pad_numericalize(entry, vocab_stoi, max_length=200) for entry in entries]


data={"train":dict(), "test":dict(), "validation":dict()}
data_splits = ["train", "test", "validation"]
for split in data_splits:
  print(f"Preprocessing {split} split")
  queries = [t for pos,t in tqdm(enumerate(msmarco[split]['query']), total=msmarco[split].num_rows) \
           for _ in range(len(msmarco[split][pos]['passages']["is_selected"]))]
  labels=[]
  docs=[]
  for pos in tqdm(range(msmarco[split].num_rows)):
      labels.extend(msmarco[split][pos]["passages"]['is_selected'])
      docs.extend(msmarco[split][pos]["passages"]['passage_text'])

  data[split]["query"]=tokenize_all(queries, pretrained_vocab.get_stoi())
  data[split]["doc"]=tokenize_all(docs, pretrained_vocab.get_stoi())
  data[split]["label"]=labels

del msmarco

train_loader = DataLoader(MsmarcoDataset(data['train']), batch_size=1, shuffle=True, drop_last=True)
val_loader   = DataLoader(MsmarcoDataset(data['validation']), batch_size=1, shuffle=True, drop_last=True)
test_loader  = DataLoader(MsmarcoDataset(data['test']), batch_size=1, shuffle=True, drop_last=True)


model=MatchPyramid(pretrained_vectors=pretrained_vectors.vectors)
# model

device='cpu'
model.to(device)


def train(model, optimizer, ep, args, threshold=0.7):
  # set the model into a training mode : the model's weights and parameters WILL BE updated!
  model.train()
  # initialize empty lists for losses and accuracies
  loss_it, acc_it = list(), list()

  # start the loop over all the training batches. This means one full epoch.
  for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s:" % (ep), total=train_loader.__len__()):
    
    batch = {'query': batch['query'].to(device), 'doc': batch['doc'].to(device), 'label': batch['label'].to(device)}

    # put parameters of the model and the optimizer to zero before doing another iteration. this prevents the gradient accumulation through batches
    optimizer.zero_grad()

    # apply the model on the batch
    logits = model(batch['query'], batch['doc'])

    loss_function = nn.BCEWithLogitsLoss()
    loss = loss_function(logits, batch['label'])

    # compute backpropagation
    loss.backward()

    # indicate to the optimizer we've done a step
    optimizer.step()

    # append the value of the loss for the current iteration (it). .item() retrieve the nuclear value as a int/long
    loss_it.append(loss.item())

    # get the predicted tags using the maximum probability from the softmax
    predicted_label = (logits>threshold)*1
    
    # Those 3 lines compute the accuracy and then append it the same way as the loss above
    correct = (predicted_label.flatten() == batch['label'].flatten()).float().sum()
    acc = correct / batch['label'].flatten().size(0)
    acc_it.append(acc.item())

  # simple averages of losses and accuracies for this epoch
  loss_it_avg = sum(loss_it)/len(loss_it)
  acc_it_avg = sum(acc_it)/len(acc_it)
  
  # print useful information about the training progress and scores on this training set's full pass (i.e. 1 epoch)
  print("Epoch %s/%s : %s : (%s %s) (%s %s)" % (colored(str(ep), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('loss', 'cyan'), loss_it_avg, colored('acc', 'cyan'), acc_it_avg))



def inference(target, loader, model, threshold=0.7):
  """
    Args:
      target (str): modify the display, usually either 'validation' or 'test'
  """

  # set the model into a evaluation mode : the model's weights and parameters will NOT be updated!
  model.eval()

  # intialize empty list to populate later on
  loss_it, acc_it, f1_it = list(), list(), list()
  # preds = predicted values ; trues = true values .... obviously~
  preds, trues = list(), list()

  # loop over the loader batches
  for it, batch in tqdm(enumerate(loader), desc="%s:" % (target), total=loader.__len__()):
    # set an environnement without any gradient. So the tensor gradients are not considered 
    with torch.no_grad():

      # put the batch to the correct device
      batch = {'query': batch['query'].to(device), 'doc': batch['doc'].to(device), 'label': batch['label'].to(device)}

      # apply the model
      logits = model(batch['query'], batch['doc'])

      loss_function = nn.BCEWithLogitsLoss()
      loss = loss_function(logits, batch['label'])

      # no need to backward() and other training stuff. Directly store the loss in the list
      loss_it.append(loss.item())

      # get the predicted tags using the maximum probability from the softmax
      predicted_label = (logits>threshold)*1
      
      # compute the accuracy and store it
      correct = (predicted_label.flatten() == batch['label'].flatten()).float().sum()
      acc = correct / batch['label'].flatten().size(0)
      acc_it.append(acc.item())
      
      # extend the predictions and true labels lists so we can compare them later on
      preds.extend(predicted_label.cpu().detach().tolist())
      trues.extend(batch['label'].cpu().detach().tolist())

  # compute the average loss and accuracy accross the iterations (batches)
  loss_it_avg = sum(loss_it)/len(loss_it)
  acc_it_avg = sum(acc_it)/len(acc_it)
  
  # print useful information. Important during training as we want to know the performance over the validation set after each epoch
  print("%s : (%s %s) (%s %s)" % ( colored(target, 'blue'), colored('loss', 'cyan'), loss_it_avg, colored('acc', 'cyan'), acc_it_avg))

  # return the true and predicted values with the losses and accuracies
  return trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it



def run_epochs(model, args):

  args['device'] =device
  # if args['cuda'] != -1:
  #     model.cuda(args['cuda'])
  #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #     args['device'] = device
  #     print("device set to %s" % (device) )

  # we set the optimizer as Adam with the learning rate (lr) set in the arguments
  # you can look at the different optimizer available here: https://pytorch.org/docs/stable/optim.html
  optimizer = optim.Adam(model.parameters(), lr = args['lr'])

  # define an empty list to store validation losses for each epoch
  val_ep_losses = list()
  # iterate over the number of max epochs set in the arguments
  for ep in range(args['max_eps']):
    # train the model using our defined function
    train(model, optimizer, ep, args)
    # apply the model for inference using our defined function
    trues, preds, val_loss_it_avg, val_acc_it_avg, val_loss_it, val_acc_it = inference("validation", val_loader, model)
    # append the validation losses (good losses should normally go down)
    val_ep_losses.append(val_loss_it_avg)

  # return the list of epoch validation losses in order to use it later to create a plot
  return val_ep_losses
    

# here you can specify if you want a GPU or a CPU by setting the cuda argument as -1 for CPU and another index for GPU. If you only have one GPU, put 0.
args={"bsize":1 }
args.update({'max_eps': 100, 'lr': 0.001, 'device': 'cpu'})
# 1e-05
print('device', device)
# Instantiate model with pre-trained glove vectors
# model = TweetModel(pretrained_embeddings, args['num_class'], args, dimension=50, freeze_embeddings = True )
model=MatchPyramid(pretrained_vectors=pretrained_vectors.vectors)
loss_list_val = run_epochs(model, args)




def plot_loss(loss_list):
  '''
  this function creates a plot. a simple curve showing the different values at each steps.
  Here we use it to plot the loss so we named it plot_loss, but the same function with different titles could be used to plot accuracies
  or other metrics for instance.
  
  Args:
    loss_list (list of floats): list of numerical values
  '''
  plt.plot(range(len(loss_list)), loss_list)
  plt.xlabel('epochs')
  # in our model we use Softmax then NLLLoss which means Cross Entropy loss
  plt.ylabel('Cross Entropy')
  # in our training loop we used an Adam optimizer so we indicate it there
  plt.title('lr: {}, optim_alg:{}'.format(args['lr'], 'Adam'))
  # let's directly show the plot when calling this function
  plt.show()

plot_loss(loss_list_val)


trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it = inference("test", test_loader, model)