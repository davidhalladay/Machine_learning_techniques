from base_import import *

from lossfunction import WMAE
from lossfunction import NAE

# setting var.
output_file_name = "Deep_structure_01.csv"
# total: 47500
train_size = 4000
valid_size = 300
feature_size = 1024
MSD_feature_size = 512
VAC_feature_size = 512

# deep learning model
class deep_model(nn.Module):
    def __init__(self, input_size = 1024 ):
        super(deep_model, self).__init__()
        self.model = nn.Sequential(
            # (batch , 1 , 1024)
            nn.Conv1d(1, 8, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(3,stride = 1,padding = 1),
            # (batch , 8 , 512)
            nn.Conv1d(8, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3,stride = 1,padding = 1),
            # (batch , 64 , 256)
            nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            # (batch , 128 , 128)
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2,stride = 2),
            # (batch , 128 , 64)

            nn.ReLU(True)
        )
        self.fc = nn.Sequential(

            nn.Linear(128 * 64, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(500, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(64, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(10, 1),
            # state = (*,1)
        )
    def forward(self, X):
        tmp = self.model(X)
        tmp = tmp.view(-1, 128 * 64)
        output = self.fc(tmp)
        return output

def save_checkpoint(checkpoint_path, model):#, optimizer):
    state = {'state_dict': model.state_dict()}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_file(file_path):
    data = np.load(file_path)['arr_0']
    return data

def dataloader(X,Y):
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_valid = X[train_size:(train_size+valid_size)]
    Y_valid = Y[train_size:(train_size+valid_size)]

    Y_train_frag = [Y_train[:,0],Y_train[:,1],Y_train[:,2]]
    Y_valid_frag = [Y_valid[:,0],Y_valid[:,1],Y_valid[:,2]]
    Y_train_frag = np.array(Y_train_frag)
    Y_valid_frag = np.array(Y_valid_frag)
    return X_train,Y_train_frag,X_valid,Y_valid_frag

def write_file(data,filename):
    data = np.array(data)
    output_file_path = os.path.join("./submission",filename)
    numofdata = data.shape[1]
    numoffeature = data.shape[0]
    file = open(output_file_path,"w")

    for row in range(numofdata):
        text = data[:,row]
        file.write("%f,%f,%f\n"%(text[0],text[1],text[2]))

    return True

def numpy_WMAE(pred_Y,gt_Y):
    # input : (3 ,num of data)
    w = [300,1,200]
    output = 0.
    if len(gt_Y.shape) == 1:
        n_sample = gt_Y.shape[0]
        n_feature = 1
        for i in range(n_sample):
            for j in range(n_feature):
                output += w[j]*np.abs(pred_Y[i] - gt_Y[i])
    else:
        n_sample = gt_Y.shape[1]
        n_feature = gt_Y.shape[0]
        for i in range(n_sample):
            for j in range(n_feature):
                output += w[j]*np.abs(pred_Y[j][i] - gt_Y[j][i])
    return output/n_sample

def numpy_NAE(pred_Y,gt_Y):
    # input : (3 ,num of data)
    output = 0.
    if len(gt_Y.shape) == 1:
        n_sample = gt_Y.shape[0]
        n_feature = 1
        for i in range(n_sample):
            for j in range(n_feature):
                output += np.abs(pred_Y[i] - gt_Y[i])/gt_Y[i]
    else:
        n_sample = gt_Y.shape[1]
        n_feature = gt_Y.shape[0]
        for i in range(n_sample):
            for j in range(n_feature):
                output += np.abs(pred_Y[j][i] - gt_Y[j][i])/gt_Y[j][i]
    return output/n_sample

def train():
    # peremeter
    aa = deep_model()
    print(aa)
    num_epochs = 40
    batch_size = 100
    learning_rate = 0.002
    learning_rate_list = np.logspace(3,15,40)

    # loading data
    print("<"+"="*50+">")
    start_time = time.time()
    X_train_path = "./data/X_train.npz"
    Y_train_path = "./data/Y_train.npz"
    X_test_path = "./data/X_test.npz"
    print("Loading dataset...")
    X_train_full = np.concatenate((load_file(X_train_path)[:,:MSD_feature_size],load_file(X_train_path)[:,5000:5000+VAC_feature_size]),axis = 1)
    Y_train_full = load_file(Y_train_path)
    X_test_full = np.concatenate((load_file(X_test_path)[:,:MSD_feature_size],load_file(X_test_path)[:,5000:5000+VAC_feature_size]),axis = 1)


    print("total training data size :",X_train_full.shape)
    print("total training label size :",Y_train_full.shape)
    print("total testing data size :",X_test_full.shape)
    X_train ,Y_train, X_valid ,Y_valid = dataloader(X_train_full,Y_train_full)
    # Random sample
    # sample_idx = random.sample([i for i in range(train_size)],5000)
    # X_train = X_train[sample_idx]
    # Y_train = Y_train[:,sample_idx].transpose()
    Y_train = Y_train.transpose()
    Y_valid = Y_valid.transpose()
    print("train data size :",X_train.shape)
    print("train label size :",Y_train.shape)
    print("valid data size :",X_valid.shape)
    print("valid label size :",Y_valid.shape)
    end_time = time.time()
    print("Processing time (sec) : %.4f" %(end_time-start_time))
    print("Loading dataset finished!")

    train_dataset = TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid),torch.from_numpy(Y_valid))

    train_dataloader = DataLoader(train_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)
    valid_dataloader = DataLoader(valid_dataset ,batch_size = batch_size ,shuffle=True ,num_workers=1)

    model_pr = deep_model()
    model_ms = deep_model()
    model_alp = deep_model()

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        model_pr = model_pr.to(device)
        model_ms = model_ms.to(device)
        model_alp = model_alp.to(device)

    optimizer_pr = optim.SGD(model_pr.parameters(), lr = learning_rate)
    optimizer_ms = optim.SGD(model_ms.parameters(), lr = learning_rate)
    optimizer_alp = optim.SGD(model_alp.parameters(), lr = learning_rate)

    #Lossfunction
    W_criterion = WMAE()
    N_criterion = NAE()

    best_loss = np.inf
    for epoch in range(num_epochs):
        model_pr.train()
        model_ms.train()
        model_alp.train()
        print("Epoch:", epoch+1)

        optimizer_pr.param_groups[0]['lr'] = 1./learning_rate_list[epoch]
        optimizer_ms.param_groups[0]['lr'] = 1./learning_rate_list[epoch]
        optimizer_alp.param_groups[0]['lr'] = 1./learning_rate_list[epoch]

        for i, (data, label) in enumerate(train_dataloader):
            batch_num = len(train_dataloader)
            optimizer_pr.zero_grad()
            optimizer_ms.zero_grad()
            optimizer_alp.zero_grad()
            data = Variable(data).to(device).unsqueeze(1)
            label = Variable(label).to(device)

            pred_pr = model_pr(data.float())
            pred_ms = model_ms(data.float())
            pred_alp = model_alp(data.float())
            pred = torch.cat((pred_pr,pred_ms,pred_alp),1)
            W_loss_pr = W_criterion(pred_pr.float(),label[:,0].float())
            W_loss_ms = W_criterion(pred_ms.float(),label[:,1].float())
            W_loss_alp = W_criterion(pred_alp.float(),label[:,2].float())
            N_loss_pr = N_criterion(pred_pr.float(),label[:,0].float())
            N_loss_ms = N_criterion(pred_ms.float(),label[:,1].float())
            N_loss_alp = N_criterion(pred_alp.float(),label[:,2].float())
            loss_pr = W_loss_pr + N_loss_pr
            loss_ms = W_loss_ms + N_loss_ms
            loss_alp = W_loss_alp + N_loss_alp

            with torch.no_grad():
                wmae = numpy_WMAE(pred.detach().numpy().transpose(),label.detach().numpy().transpose())
                nae = numpy_NAE(pred.detach().numpy().transpose(),label.detach().numpy().transpose())

            W_loss = W_loss_pr.item() + W_loss_ms.item() + W_loss_alp.item()
            N_loss = N_loss_pr.item() + N_loss_ms.item() + N_loss_alp.item()

            loss_pr.backward()
            loss_ms.backward()
            loss_alp.backward()
            optimizer_pr.step()
            optimizer_ms.step()
            optimizer_alp.step()

            print('Epoch [%d/%d], Iter [%d/%d] W_loss %.4f N_loss %.4f , LR = %.6f'
                    %(epoch, num_epochs, i+1, len(train_dataloader), wmae ,nae , optimizer_pr.param_groups[0]['lr']))

        print("<"+"="*50+">")
        model_pr.eval()
        model_ms.eval()
        model_alp.eval()
        W_total_loss = 0.
        N_total_loss = 0.
        with torch.no_grad():
            for i, (data, label) in enumerate(valid_dataloader):
                batch_num = len(valid_dataloader)
                data = Variable(data).to(device).unsqueeze(1)
                label = Variable(label).to(device)
                pred_pr = model_pr(data.float())
                pred_ms = model_ms(data.float())
                pred_alp = model_alp(data.float())
                pred = torch.cat((pred_pr,pred_ms,pred_alp),1)
                W_loss_pr = W_criterion(pred_pr.float(),label[:,0].float())
                W_loss_ms = W_criterion(pred_ms.float(),label[:,1].float())
                W_loss_alp = W_criterion(pred_alp.float(),label[:,2].float())
                N_loss_pr = N_criterion(pred_pr.float(),label[:,0].float())
                N_loss_ms = N_criterion(pred_ms.float(),label[:,1].float())
                N_loss_alp = N_criterion(pred_alp.float(),label[:,2].float())

                W_loss = W_loss_pr.item() + W_loss_ms.item() + W_loss_alp.item()
                N_loss = N_loss_pr.item() + N_loss_ms.item() + N_loss_alp.item()

                wmae = numpy_WMAE(pred.cpu().numpy().transpose(),label.cpu().numpy().transpose())
                nae = numpy_NAE(pred.cpu().numpy().transpose(),label.cpu().numpy().transpose())
                W_total_loss += wmae
                N_total_loss += nae

            print('Epoch [%d/%d],  W_loss %.4f N_loss %.4f , LR = %.6f'
                %(epoch+1, num_epochs,  W_total_loss/batch_num, N_total_loss/batch_num, optimizer_pr.param_groups[0]['lr']))
            print("<"+"="*50+">")

            # valid save
            v_X_valid = torch.from_numpy(X_valid).to(device)
            v_X_valid = v_X_valid.unsqueeze(1)
            pred_pr = model_pr(v_X_valid.float())
            pred_ms = model_ms(v_X_valid.float())
            pred_alp = model_alp(v_X_valid.float())
            output = torch.cat((pred_pr,pred_ms,pred_alp),1)
            output = output.cpu().numpy().transpose()
            write_file(output,"Deep_structure_valid")
            gt_valid = Y_valid.transpose()
            wmae = numpy_WMAE(output,gt_valid)
            nae = numpy_NAE(output,gt_valid)
            print("valid file done! W_loss %.4f N_loss %.4f "%(wmae,nae))
            print("<"+"="*50+">")
            if (wmae + nae) < best_loss:
                # Testing session
                best_loss = wmae + nae
                best_W_loss = wmae
                best_N_loss = nae
                print("[Find best!]")
                X_test = torch.from_numpy(X_test_full).to(device)
                X_test = X_test.unsqueeze(1)
                pred_pr = model_pr(X_test.float())
                pred_ms = model_ms(X_test.float())
                pred_alp = model_alp(X_test.float())
                output = torch.cat((pred_pr,pred_ms,pred_alp),1)
                output = output.detach().numpy().transpose()
                write_file(output,output_file_name)
                print("")
                print("Testing file done!")

                print("<"+"="*28+"Done"+"="*28+">")
    print("Best : W_loss = %.4f, N_loss = %.4f"%(best_W_loss,best_N_loss))
    print("<"+"="*50+">")
    return 0
if __name__ == "__main__":
    train()
