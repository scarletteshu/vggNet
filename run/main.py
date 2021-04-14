import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from models_init import *
from dataLoader import *
from recordAndPlot import *


if __name__ == "__main__":
    # load configs
    f = open("../configs/config.yaml", 'r', encoding='utf-8')
    data = f.read()
    datadict = yaml.load(data, Loader=yaml.FullLoader) # str -> dict/list
    vgg_cfgs = datadict['VGG']
  
    # load data
    dl = Dataloader(vgg_cfgs, "../data/train/", "../data/test/")
    tr_datasets, tr_dataloader = dl.trDataLoader()
    tst_datasets, tst_dataloader = dl.tstDataLoader()
    
    # model->CUDA->resume->parallel
    model = VGG_init(vgg_cfgs['structure'], 'vgg16')
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # cuda before optim

    optimizer = optim.Adam([{'params':model.parameters(),'initial_lr':float(vgg_cfgs['lr'])}], float(vgg_cfgs['lr']))  #'initial_lr' is needed
    criterion = nn.CrossEntropyLoss() #default reduction = 'mean'
    
    # checkcpoint resume
    if os.path.isfile("../results/vgg16.pth.tar"):
        checkpoint = torch.load("vgg16.pth.tar")
        model.load_state_dict(checkpoint['model_state_dict'])
        # load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        start_epoch = checkpoint['start_epoch']
        loss = checkpoint['loss']
    else:
        start_epoch = 0

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, float(vgg_cfgs['sch_gamma']), last_epoch= start_epoch)
    # print("schdu lr:{}, optim lr:{}\n".format(scheduler.get_lr()[0], optimizer.param_groups[0]['lr']))
    print(model)
    for epoch in range(start_epoch+1, vgg_cfgs['EPOCH']):
        model.train()
        for i, (img, label) in enumerate(tr_dataloader):
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)
            acc = sum(torch.max(out, 1)[1] == label)/label.shape[0] 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch:{}, iter:{}, train_loss per img:{}, train_acc:{}'.format(epoch, i, loss, acc))
            # save results
            record = Record(loss, acc)
            record.save(epoch, "/vgg16/train/")
        # decay lr
        if epoch % 10 == 0:
            print("lr scheduler step!, lr:{}".format(scheduler.get_lr()[0]))
            scheduler.step()
        # save model
        torch.save(
            {
                'start_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            "../results/vgg16.pth.tar"
        )
        
        # model eval
        with torch.no_grad():
            model.eval()
            tst_loss = 0
            tst_acc = 0
            itertimes = len(tst_datasets) / vgg_cfgs['batch_size']

            for tst_img, tst_label in tst_dataloader:
                tst_img, tst_label = tst_img.to(device), tst_label.to(device)
                tst_out = model(tst_img)
                tst_loss = tst_loss + criterion(tst_out, tst_label)
                tst_acc = tst_acc + sum(torch.max(tst_out, 1)[1] == tst_label)/tst_label.shape[0]
            print("epoch:{}, tst_loss:{}, tst_acc:{}".format(epoch, tst_loss/itertimes, tst_acc/itertimes))

            record = Record(tst_loss/itertimes, tst_acc/itertimes)
            record.save(epoch, "./vgg16/test/")










