import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
from torch import relu
from torch import optim,nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import PIL
import time

torch.manual_seed(1)

labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

root='data'
label=True

while label:
    if label==None:
        break
    label=str(input('LETTER: '))
    cv2.namedWindow('CAPTURE DATA '+label.upper())
    vc=cv2.VideoCapture(0)
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False

    for iter in range(60):
        if rval:
            image_name=os.path.join(root,label.upper(),str(iter)+'.jpg')
            time.sleep(.5)
            rval,frame=vc.read()
            frame=cv2.resize(frame,(480,480))
            cv2.imshow('CAPTURE DATA'+label.upper(),frame)
            cv2.imwrite(image_name,frame)
            print(cv2.imwrite(image_name,frame))
            key=cv2.waitKey(20)
            if key == 27:
                break
        if key==27:
            break
    cv2.destroyAllWindows()
    if key==27:
        break

class data(Dataset):
    def __init__(self):
        self.y=[]
        self.x=[]
        for label in range(len(labels)):
            images=os.listdir(root+'/'+labels[label])
            for i in range(len(images)):
                self.y.append(label)
                images[i]=root+'/'+labels[label]+'/'+images[i]
            self.x+=images
    def __getitem__(self,index):
        x=PIL.Image.open(self.x[index])
        x=x.convert('RGB')
        x=np.asarray(x)
        x=np.transpose(x,(2,0,1))
        x=torch.tensor(x)
        x=x.type(torch.float32)
        x=torch.div(x,255)
        x=x.view(3,480,480)
        y=self.y[index]
        y=np.asarray(y)
        y=torch.tensor(y,dtype=torch.long)
        return x,y
    def __len__(self):
        return(len(self.y))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.dropout=nn.Dropout(0.1)
        self.conv1=nn.Conv2d(3,16,3)
        self.conv2=nn.Conv2d(16,32,3)
        self.conv3=nn.Conv2d(32,64,3)
        self.conv4=nn.Conv2d(64,128,3)
        self.max_pool=nn.MaxPool2d(3)
        self.linear1=nn.Linear(2048,2048)
        self.linear2=nn.Linear(2048,1024)
        self.linear3=nn.Linear(1024,25)
    def forward(self,x):
        x=self.conv1(x)
        x=torch.relu(x)
        x=self.max_pool(x)
        x=self.conv2(x)
        x=torch.relu(x)
        x=self.max_pool(x)
        x=self.conv3(x)
        x=torch.relu(x)
        x=self.max_pool(x)
        x=self.conv4(x)
        x=torch.relu(x)
        x=self.max_pool(x)
        x=x.view(x.size(0),-1)
        x=self.linear1(x)
        x=self.dropout(x)
        x=torch.relu(x)
        x=self.linear2(x)
        x=self.dropout(x)
        x=torch.relu(x)
        x=self.linear3(x)
        return(x)

dataset=data()
print(len(dataset))
train_data,val_data=random_split(dataset,[925,200])

train_loader=DataLoader(train_data,batch_size=45,shuffle=True)
val_loader=DataLoader(val_data,batch_size=5,shuffle=True)

def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')

model=Net()
model.apply(weight_init)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0)
useful_dict={'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}

epochs=75
for epoch in range(epochs):
    print("<<<<<<<<<<<<<<<EPOCH: ",epoch+1,'>>>>>>>>>>>>>>>')
    model.train()
    train_loss=0
    correct=0
    for i,(x,y) in enumerate(train_loader):
        print('Batch: ',i+1)
        optimizer.zero_grad()
        z=model(x)
        loss=criterion(z,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.data.item())
        
        train_loss+=loss.data.item()
        _,label=torch.max(z,1)
        correct+=(label==y).sum().item()
    train_loss/=(i+1)
    train_acc=100*correct/len(train_data)
    print("EPOCH LOSS: ",train_loss)
    print('EPOCH ACCURACY: ',train_acc)
    useful_dict['train_loss'].append(train_loss)
    useful_dict['train_acc'].append(train_acc)
    
    model.eval()
    val_loss=0
    correct=0
    for i,(x,y) in enumerate(val_loader):
        z=model(x)
        loss=criterion(z,y)
        val_loss+=loss.data.item()
        _,label=torch.max(z,1)
        correct+=(label==y).sum().item()
    val_loss/=(i+1)
    val_acc=100*correct/len(val_data)
    print("VALIDATION LOSS: ",val_loss)
    print("VALIDATION ACCURACY",val_acc)
    useful_dict['val_loss'].append(val_loss)
    useful_dict['val_acc'].append(val_acc)

    torch.save(model.state_dict(),'C:/Users/waiku/Desktop/Python/Projects/Machine Learning/RealTimeASL/models'+'/model'+str(epoch))

max_acc=[max(useful_dict['val_acc'])]*len(useful_dict['val_acc'])
min_loss=[min(useful_dict['val_loss'])]*len(useful_dict['val_loss'])
plt.plot(useful_dict['val_loss'],label='Validation Loss')
plt.plot(useful_dict['train_loss'],label="Train Loss")
plt.plot(min_loss,label='Minimum Loss')
plt.legend()
plt.show()
plt.plot(useful_dict['val_acc'],label="Validation Accuracy")
plt.plot(useful_dict['train_acc'],label="Train Accuracy")
plt.plot(max_acc,label='Maximum Accuracy')
plt.legend()
plt.show()
print("MAX VALIDATION ACCURACY: ",max(useful_dict['val_acc']))
print("MINIMUM VALIDATION LOSS: ",min(useful_dict['val_loss']))

model=Net()
model.load_state_dict(torch.load('models'+'/model'+str(18)))

cv2.namedWindow('preview')
vc=cv2.VideoCapture(0)

if vc.isOpened():
    rval,show_frame=vc.read()
    label=None
else:
    rval=False

while rval:
    show_frame=cv2.resize(show_frame,(480,480))
    np_frame=np.asarray(show_frame)
    tensor_frame=torch.from_numpy(np_frame)
    tensor_frame=tensor_frame.type(torch.float32)
    tensor_frame=torch.reshape(tensor_frame,(1,3,480,480))
    z=model(tensor_frame)
    past_label=label
    _,label=torch.max(z,1)
    show_frame=cv2.putText(show_frame,str(labels[int(label.item())]),(50,70),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),3)
    cv2.imshow('preview',show_frame)
    rval,show_frame=vc.read()
    key=cv2.waitKey(20)
    if key == 27:
        break
cv2.destroyWindow('preview')
