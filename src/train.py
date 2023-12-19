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
    bqtt = 512

    def get_batch(source, i):
        seq_len = min(bqtt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        return torch.from_numpy(data).to("cuda:0").transpose(0, 1) if len(data.shape)>2 else torch.from_numpy(data).to("cuda:0")

    model = TransformerModel(input_size=52,
                              nlen=nlen, 
                              hidden_size=256, 
                              num_layers=4, 
                              num_heads=4, 
                              output_size=34).to("cuda:0")

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    criterion = nn.CrossEntropyLoss()
    def train():
        # Turn on training mode which enables dropout.
        model.train()
        
        bar = tqdm(enumerate(range(0, x_train.shape[0] - 1, bqtt)))

        for batch, i in bar:

            data = get_batch(x_train, i).to(torch.float32)
            targets = get_batch(y_train, i).to(torch.float32)
            sizes = get_batch(x_len_train, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            output = model(data, sizes)
            loss = criterion(output, targets)
            res = output>0.5
            true_true = res[targets==1].sum() / res.sum()
            true_false = ((~res)[targets==0]).sum() / (~res).sum()
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
                data = get_batch(x_test, i).to(torch.float32)
                targets = get_batch(y_test, i).to(torch.float32)
                sizes = get_batch(x_len_test, i)
                output= model(data, sizes)
                loss = criterion(output, targets)
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