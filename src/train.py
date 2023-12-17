import torch
import torch.nn as nn
import time
import math
from generate_train_data import generate_train_test_local
from transformer  import TransformerModel
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


if __name__ == "__main__":
    print(torch.cuda.is_available())
    # x_train, x_test, y_train, y_test = generate_train_test()
    x_train, x_test, x_len_train, x_len_test, y_train, y_test = generate_train_test_local()
    ntokens = x_train.shape[-1]
    nlen = x_train.shape[1]
    print(x_train.shape)
    print(y_train.shape)
    bqtt = 128

    def get_batch(source, i):
        seq_len = min(bqtt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        return data.to("cuda:0").transpose(0, 1) if data.dim()>1 else data.to("cuda:0")

    model = TransformerModel(ntokens, nlen, 600, 4, 600, 4).to("cuda:0")
    # Define the binary cross-entropy loss
    # criterion = nn.CrossEntropyLoss()
    
    # criterion = nn.NLLLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    def bce(pred, label):
        return (-label * torch.log(pred) - (1 - label) * torch.log(1 - pred)).sum(-1).mean()
    
    criterion = nn.BCELoss()

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        
        bar = tqdm(enumerate(range(0, x_train.size(0) - 1, bqtt)))

        for batch, i in bar:
            data = get_batch(x_train, i)
            targets = get_batch(y_train, i)
            sizes = get_batch(x_len_train, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            output, mask = model(data, sizes)
            # output = output.view(-1, 34)
            mask[:20] = False
            ones = targets.any(-1)
            ones = mask & (ones)
            all_one = targets[ones]==1
            uv = torch.stack(torch.meshgrid(torch.arange(ones.sum()), torch.arange(34)), dim=-1).to("cuda:0")
            zeros = uv[~all_one]
            chosen = zeros[(torch.rand(all_one.sum(), device="cuda:0") * zeros.shape[0]).to(torch.long)]
            all_one[chosen[:, 0], chosen[:, 1]] = True
            # loss1 = criterion(output[mask], targets[mask])
            # loss2 = criterion(output[targets==1], targets[targets==1])
            # loss = loss1 + loss2
            # loss = criterion(output[mask].reshape(-1), targets[mask].reshape(-1))
            loss = criterion(output[ones][all_one], targets[ones][all_one])
            # loss = criterion(output[targets==1], targets[targets==1])
            res = output[ones][all_one] > 0.5
            gt_true = targets[ones][all_one] > 0
            # uncorrect = res != (targets.any(-1))[mask]
            # false_true = (res & uncorrect).sum() / (targets.any(-1))[mask].sum()
            # false_false = ((~res) & uncorrect).sum() / (~(targets.any(-1))[mask]).sum()
            true_true = res[gt_true].sum() / res.sum()
            true_false = ((~res)[~gt_true]).sum() / (~res).sum()
            loss.backward()
            optimizer.step()

            # total_loss += loss.item()
            # bar.set_description(f'Epoch [{epoch+1}/{batch}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')
            bar.set_description(f'Epoch [{epoch+1}/{batch}], Loss: {loss.item():.4f}, tt: {true_true: .2f}, tf: {true_false: .2f}')
        bar.close()
    def evaluate(x_test, y_test):
    # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = x_test.shape[1]
        with torch.no_grad():
            for i in range(0, x_test.shape[0] - 1, bqtt):
                data = get_batch(x_test, i)
                targets = get_batch(y_test, i)
                sizes = get_batch(x_len_test, i)
                output, mask = model(data, sizes)
                mask[:8] = 0
                loss = criterion(output[mask], (targets)[mask])
                total_loss += len(data) * loss.item()
        return total_loss / (len(x_test) - 1)

    try:
        best_val_loss = False
        for epoch in range(1, 40):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(x_test, y_test)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} |'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            
            if not best_val_loss or val_loss < best_val_loss:
                with open("./model/model2.ckpt", 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                optimizer.param_groups[0]["lr"] /= 1.1

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')