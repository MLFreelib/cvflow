import os

import numpy as np
import torch
from common.prepare_plates_data import *
from models.crnn import CRNN
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop, resize, rgb_to_grayscale

from ultralytics import YOLO
import torch.optim as optim
from torch.nn import CTCLoss
from common.ctc_decoder import ctc_decode


class PlatesCropsDataset(Dataset):
    CHARS = '0123456789АВЕКМНОРСТУХ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    CHARS_ENRU = dict(zip(list('ABEKMHOPCTYX'), list('АВЕКМНОРСТУХ')))

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width


    def __len__(self):
        return len(self.paths)

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        img_folder = os.path.join(root_dir, 'crops', mode)
        for path in os.listdir(img_folder):
            annot_path =  os.path.join(root_dir, 'annotations', mode, path[:path.find('.')]+ '.annot')
            try:
                with open(annot_path, 'r') as fr:
                    text = fr.read().replace('\n', '').strip().upper()
                    res_text = []
                    for char in list(text):
                        if char in self.CHARS_ENRU:
                            res_text.append(self.CHARS_ENRU[char])
                        else:
                            res_text.append(char)

                    mapping[path] = ''.join(res_text)
            except FileNotFoundError:
                print('Corrupted annotation for {}'.format(annot_path))

        paths, texts = zip(*mapping.items())
        paths = [os.path.join(img_folder, p) for p in paths]
        return paths, texts

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path)  # grey-scale
        except IOError:
            #print('Corrupted image for %d' % index)
            return self[index + 1]

        image = torch.tensor(np.array(image)).permute(2, 0, 1)
        image = rgb_to_grayscale(image)
        image = resize(image, (self.img_height, self.img_width)).unsqueeze(0) # bilinear resize
        #image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        #image = np.array(image)
        image = image.view((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        # image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image
        

def data_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


def train_crnn_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()


def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation


class PlatesTrainingPipeline:
    def __init__(self, yolo_config: dict, crnn_config: dict,
                 datasets_folder='datasets',
                 dataset_name='russian_plates', **kwargs):
        
        self.datasets_folder = datasets_folder
        self.dataset_name = dataset_name
        save_folder = os.path.join(datasets_folder, dataset_name)
        if not self._check_data_is_generated(save_folder):
            # generate data
            prepare_russian_plates_bboxes(save_folder=save_folder)
            
        if not self._check_yolo_config_is_generated(dataset_name):
            # generate config for training yolo model
            self.generate_yolo_config(dataset_name)
        self.config = dataset_name + '.yaml'
        self.yolo_train_config = yolo_config
        self.crnn_train_config = crnn_config
        # pretrained yolo model name
        self.yolo_model_name = yolo_config.get('yolo_model_name', 'yolov8s.pt')
        self.yolo_model = YOLO(self.yolo_model_name)
        
    def _check_yolo_config_is_generated(self, dataset_name):
        return os.path.exists(dataset_name + '.yaml')
    
    def _check_data_is_generated(self, save_folder):
        return os.path.exists(os.path.join(save_folder, 'crops'))\
            and os.path.exists(os.path.join(save_folder, 'labels'))\
            and os.path.exists(os.path.join(save_folder, 'annotations'))\
            and os.path.exists(os.path.join(save_folder, 'images'))
    
    def generate_yolo_config(self, dataset_name, cls_name='plate'):
        with open(dataset_name + '.yaml', 'w') as f:
            f.write(f"""path: {dataset_name}\ntrain: images/train\ntest: images/test\nval: images/val\nnc: 1\nnames: ['{cls_name}']\n""")
    
    def train(self):
        # FIRST STAGE – train YOLO v8 model (small)
        
        self.yolo_model.train(data=self.config, **self.yolo_train_config)
        
        # SECOND STAGE – train CRNN model
        epochs = self.crnn_train_config['epochs']
        train_batch_size = self.crnn_train_config['train_batch_size']
        eval_batch_size = self.crnn_train_config['eval_batch_size']
        lr = self.crnn_train_config['lr']
        show_interval = self.crnn_train_config['show_interval']
        valid_interval = self.crnn_train_config['valid_interval']
        save_interval = self.crnn_train_config['save_interval']
        cpu_workers = self.crnn_train_config['cpu_workers']
        reload_checkpoint = self.crnn_train_config['reload_checkpoint']
        valid_max_iter = self.crnn_train_config['valid_max_iter']

        img_width = self.crnn_train_config['img_width']
        img_height = self.crnn_train_config['img_height']
        data_dir = os.path.join(self.datasets_folder, self.dataset_name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {device}')

        train_dataset = PlatesCropsDataset(root_dir=data_dir, mode='train',
                                        img_height=img_height, img_width=img_width)
        valid_dataset = PlatesCropsDataset(root_dir=data_dir, mode='dev',
                                        img_height=img_height, img_width=img_width)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            #num_workers=cpu_workers,
            collate_fn=data_collate_fn)
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=eval_batch_size,
            shuffle=True,
            #num_workers=cpu_workers,
            collate_fn=data_collate_fn)

        num_class = len(PlatesCropsDataset.LABEL2CHAR) + 1
        crnn = CRNN(1, img_height, img_width, num_class,
                    map_to_seq_hidden=self.crnn_train_config['map_to_seq_hidden'],
                    rnn_hidden=self.crnn_train_config['rnn_hidden'],
                    leaky_relu=self.crnn_train_config['leaky_relu'])
        if reload_checkpoint:
            crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
            
        crnn.to(device)

        optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
        criterion = CTCLoss(reduction='sum', zero_infinity=True)
        criterion.to(device)

        assert save_interval % valid_interval == 0
        i = 1
        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')
            tot_train_loss = 0.
            tot_train_count = 0
            for train_data in train_loader:
                loss = train_crnn_batch(crnn, train_data, optimizer, criterion, device)
                train_size = train_data[0].size(0)

                tot_train_loss += loss
                tot_train_count += train_size
                if i % show_interval == 0:
                    print('train_batch_loss[', i, ']: ', loss / train_size)

                if i % valid_interval == 0:
                    evaluation = evaluate(crnn, valid_loader, criterion,
                                        decode_method=self.crnn_train_config['decode_method'],
                                        beam_size=self.crnn_train_config['beam_size'])
                    print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

                    if i % save_interval == 0:
                        prefix = 'crnn'
                        loss = evaluation['loss']
                        save_model_path = os.path.join(self.crnn_train_config['checkpoints_dir'],
                                                    f'{prefix}_{i:06}_loss{loss}.pt')
                        torch.save(crnn.state_dict(), save_model_path)
                        print('save model at ', save_model_path)

                i += 1

            print('train_loss: ', tot_train_loss / tot_train_count)
