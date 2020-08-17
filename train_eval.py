import os
import copy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from config import logger, Config
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    # topk(1, dim=1, largest=True, sorted=True)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    # print('correct: ' + str(correct))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
       
        tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']], 
                max_length=opt.max_length, 
                data_file='./embedding/{0}_{1}_tokenizer.dat'.format(opt.model_name, opt.dataset),
                )
        embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab, 
                embed_dim=opt.embed_dim, 
                data_file='./embedding/{0}_{1}d_{2}_embedding_matrix.dat'.format(opt.model_name, str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, target_dim=self.opt.polarities_dim)
        testset = SentenceDataset(opt.dataset_file['test'], tokenizer, target_dim=self.opt.polarities_dim)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)   # , drop_last=True
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]

                outputs = self.model(inputs)                            # torch.Size([16, 20, 4])
                targets = sample_batched['label'].to(self.opt.device)   # torch.Size([16, 20])

                loss = 0
                # acc = 0

                if self.opt.model_name == 'textcnn':
                    for idx, _ in enumerate(Config.label_names):
                        loss += criterion(outputs[:, idx, :], targets[:, idx]) / len(Config.label_names)
                        # acc += accuracy(outputs[:, idx, :], targets[:, idx]) / len(label_names)
                    loss.backward()
                    optimizer.step()
                    if global_step % self.opt.log_step == 0:
                        for idx, _ in enumerate(Config.label_names):
                            # n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                            # n_total += len(outputs)
                            n_correct += (torch.argmax(outputs[:, idx, :], -1) == targets[:, idx]).sum().item()
                            n_total += len(outputs)
                        train_acc = n_correct / n_total
                        # test_acc, f1 = self._evaluate()
                        test_acc = self._evaluate_textcnn()
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = './state_dict/{0}_{1}_{2}class_acc{3:.4f}'.format(self.opt.model_name, self.opt.dataset, self.opt.polarities_dim, test_acc)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(path))
                        # if f1 > max_f1:
                        #     max_f1 = f1
                        # logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
                        logger.info('global_step:{}, loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}'.format(global_step, loss.item(), train_acc, test_acc))
                
                elif self.opt.model_name == 'gcae':
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    if global_step % self.opt.log_step == 0:    # 每隔opt.log_step就输出日志
                        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                        n_total += len(outputs)
                        train_acc = n_correct / n_total
                        test_acc, f1 = self._evaluate_gcae()
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = './state_dict/{0}_{1}_{2}class_acc{3:.4f}'.format(self.opt.model_name, self.opt.dataset, self.opt.polarities_dim, test_acc)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(path))
                        if f1 > max_f1:
                            max_f1 = f1
                        logger.info('global_step:{}, loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(global_step, loss.item(), train_acc, test_acc, f1))

        return max_test_acc, max_f1, path
    
    def _evaluate_textcnn(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_dataloader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.opt.device)
        
                t_outputs = self.model(t_inputs)

                for idx, _ in enumerate(Config.label_names):
                    n_test_correct += (torch.argmax(t_outputs[:, idx, :], -1) == t_targets[:, idx]).sum().item()
                    n_test_total += len(t_outputs)
                
        test_acc = n_test_correct / n_test_total
        return test_acc

    def _evaluate_gcae(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_dataloader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)
                
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0) if t_targets_all is not None else t_targets
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0) if t_outputs_all is not None else t_outputs
        
        labels = t_targets_all.data.cpu()
        predic = torch.argmax(t_outputs_all, -1).cpu()
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(labels, predic, labels=[0, 1, 2, 3], average='macro')

        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion

        return test_acc, f1

    def _test(self, model_path):
        # test
        # self.model.load_state_dict(torch.load(model_path))
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion  = self._evaluate_gcae(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)
        # logger.info('f1: {:.4f},'.format(f1))

    def run(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        self._reset_params()
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer)
        logger.info('max_test_acc: {:.4f}, max_f1: {:.4f}'.format(max_test_acc, max_f1))
        # logger.info('max_test_acc: {:.4f}'.format(max_test_acc))
        logger.info('#' * 60)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        if self.opt.model_name == 'gcae':
            self._test(model_path)
