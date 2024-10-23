import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# I imputted the completed function in the official tutorial for training my model, and I asked chatGPT to help me compute the accuracy.

def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    
    custom_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.FashionMNIST('./data', train = True, download=True, transform=custom_transform)
    test_set = datasets.FashionMNIST('./data', train = False, transform=custom_transform)
    
    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    return loader


def build_model():
    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    return model


def train_model(model, train_loader, criterion, T):
    # set the model to train mode before iterating
    model.train()
    counter = 0
    # outer for loop: iterates through epochs
    for epoch in range(T):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # inner for loop: iterates through (images, labels) pairs from the train_loader
        for (images, labels) in train_loader:
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            total_predictions += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability (ChatGPT helped me with this part)
            correct_predictions += (predicted == labels).sum().item()  # Update correct count
            accuracy_percent = round((correct_predictions / total_predictions) * 100, 2)
        # printing everything
        print(f'Train Epoch: {counter}   Accuracy: {correct_predictions}/{total_predictions}({accuracy_percent}%)   Loss: {round(running_loss / len(train_loader), 3)}')
        counter += 1



def evaluate_model(model, test_loader, criterion, show_loss = True):
    correct = 0
    total = 0
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            
            # run images through the model to calculate the outputs
            outputs = model(images)
            
            # we want the class with the highest energy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # calculate the loss and update the total
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        accuracy = (correct / total) * 100
        if (show_loss == True):
            print(f'Average Loss: {round(running_loss / len(test_loader), 4)}')
        print(f'Accuracy: {round(accuracy, 2)}%')


def predict_label(model, test_images, index):
    # find logits
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)
    
    # assumed class names
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    index = 0
    sorted_with_index = []
    for p in prob[0]:
        sorted_with_index.append([p.item(), index])
        index = index + 1
    sorted_with_index = sorted(sorted_with_index, reverse = True)
    top1 = sorted_with_index[0]
    top1_label = class_names[top1[1]]
    top2 = sorted_with_index[1]
    top2_label = class_names[top2[1]]
    top3 = sorted_with_index[2]
    top3_label = class_names[top3[1]]
    print(f'{top1_label}: {round(top1[0] * 100, 2)}%')
    print(f'{top2_label}: {round(top2[0] * 100, 2)}%')
    print(f'{top3_label}: {round(top3[0] * 100, 2)}%')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
