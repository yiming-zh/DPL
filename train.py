import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

from model import RGATABSA, MultiBERT, MultiBERT_DPL


class ABSATrainer(object):
    def __init__(self, args, emb_matrix=None, pre_trained=None):
        self.lr_cut = 1
        self.args = args
        self.step = self.args.step_long
        self.emb_matrix = emb_matrix
        if self.args.bert_for_sentence == 1:
            self.model = MultiBERT(args)
        elif self.args.bert_for_sentence == 2:
            self.model = MultiBERT_DPL(args)
        else:
            self.model = RGATABSA(args)
        # if pre_trained is not None:
        #     self.pre_trained_load(pre_trained)
        self.parameters = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.model.cuda()
        self.optimizer_generate()

    def optimizer_generate(self):
        # optimizer for decoder
        self.de_optimizer = torch.optim.Adam(
            self.model.decoder.parameters(),
            lr=self.args.de_lr/self.lr_cut,
            weight_decay=self.args.l2
        )

        # optimizer for encoder
        bert_model = self.model.encoder.encoder.Sent_encoder
        bert_params_dict = list(map(id, bert_model.parameters()))
        encoder_params_no_bert = filter(
            lambda p: id(p) not in bert_params_dict,
            self.model.encoder.parameters()
        )
        encoder_parameters = [
            {"params": encoder_params_no_bert},
            {
                "params": bert_model.parameters(),
                "lr": self.args.bert_lr/self.lr_cut
            }
        ]
        self.en_optimizer = torch.optim.Adam(
            encoder_parameters,
            lr=self.args.en_lr/self.lr_cut,
            weight_decay=self.args.l2
        )

    def zero_grad(self):
        self.en_optimizer.zero_grad()
        self.de_optimizer.zero_grad()

    def lr_update(self):
        self.lr_cut *= 2
        self.optimizer_generate()

    # load model_state and args
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint["model"])
        self.args = checkpoint["config"]

    # save model_state and args
    def save(self, filename):
        params = {
            "model": self.model.state_dict(),
            "config": self.args,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def pre_trained_load(self, pre_model_path):
        pretrainer = torch.load(pre_model_path)
        pretrained_dict = pretrainer.model.state_dict()
        model_dict = self.model.state_dict()
        # 载入对应存在的参数
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def update(self, batch, level=None):
        self.level = level
        inputs, label = self.batch_generate(batch)

        # step forward
        self.model.train()
        inputs, sentence_label = self.batch_divide(inputs)
        # while True:
        loss, acc = self.train_aspect_and_sentence(
            inputs, label, sentence_label
        )
        # return loss, acc
        G_loss, D_loss, noise_prob = self.train_anti(
            # inputs, label
            inputs, sentence_label
        )
            # print(loss, acc, G_loss, D_loss, noise_prob)

        return (
            G_loss,
            D_loss,
            loss,
            noise_prob
        ), acc

    def train_aspect(self, inputs, label):
        self.zero_grad()
        logits, sentence_logits, noise_logits = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction="mean")
        # backward
        loss.backward()
        self.en_optimizer.step()
        self.de_optimizer.step()
        acc = self.acc_sum(logits, label) / label.size()[0]
        return loss.data, acc

    def train_sentence(self, inputs, label, sentence_label):
        self.zero_grad()
        logits, sentence_logits, noise_logits = self.model(inputs)
        loss = F.cross_entropy(
            sentence_logits, sentence_label, reduction="mean"
        )
        # backward
        loss.backward()
        self.en_optimizer.step()
        self.de_optimizer.step()
        acc = self.acc_sum(logits, label) / label.size()[0]
        sentence_acc = self.acc_sum(
            sentence_logits, sentence_label
        ) / sentence_label.size()[0]

        return loss, (acc, sentence_acc)

    def train_aspect_and_sentence(self, inputs, label, sentence_label):
        self.zero_grad()
        logits, sentence_logits, noise_logits = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction="mean")
        sentence_loss = F.cross_entropy(
            sentence_logits, sentence_label, reduction="mean"
        )
        if self.level == 'aspect':
            sentence_loss = sentence_loss * self.args.pseudo_weight
        elif self.level == 'sentence':
            loss = loss * self.args.pseudo_weight

        total_loss = loss + sentence_loss * self.args.sentence_weight
        # backward
        total_loss.backward()
        self.en_optimizer.step()
        self.de_optimizer.step()
        acc = self.acc_sum(logits, label) / label.size()[0]
        sentence_acc = self.acc_sum(
            sentence_logits, sentence_label
        ) / sentence_label.size()[0]

        return loss.data, (acc, sentence_acc)

    def train_anti(self, inputs, anti_label):
        # multi_sentence_label = self.label2multi_label(sentence_label)
        multi_sentence_label = self.label2multi_label(anti_label)
        # import ipdb; ipdb.set_trace()
        D_loss, noise_prob = self.train_real_loss(
            inputs, multi_sentence_label
        )

        G_loss, _ = self.train_fake_loss(inputs, multi_sentence_label)

        self.model.train()
        return \
            G_loss.data,\
            D_loss.data,\
            noise_prob.mean(dim=0).detach().cpu().numpy()

    def train_real_loss(self, inputs, multi_sentence_label):
        # freeze encoder
        self.model.encoder.eval()
        self.model.decoder.train()
        self.de_optimizer.zero_grad()
        logits, sentence_logits, noise_logits \
            = self.model(inputs)
        noise_prob = torch.sigmoid(noise_logits)
        real_loss = F.binary_cross_entropy(
            noise_prob, multi_sentence_label, reduction="mean"
        )
        D_loss = real_loss * self.args.anti_weight
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.decoder.parameters(), self.args.clip
        )
        self.de_optimizer.step()
        return D_loss, noise_prob

    def train_fake_loss(self, inputs, multi_sentence_label):
        # freeze decoder
        self.model.decoder.eval()
        self.model.encoder.train()
        # freeze bert
        # self.model.encoder.encoder.Sent_encoder.eval()
        self.en_optimizer.zero_grad()
        logits, sentence_logits, noise_logits \
            = self.model(inputs)
        noise_prob = torch.sigmoid(noise_logits)
        # fake_loss = F.binary_cross_entropy(
        #     noise_prob, (1-multi_sentence_label), reduction="mean"
        # )
        fake_loss = - F.binary_cross_entropy(
            (1-noise_prob), (1-multi_sentence_label), reduction="mean"
        )
        G_loss = fake_loss * self.args.anti_weight
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.encoder.parameters(), self.args.clip
        )
        self.en_optimizer.step()
        return G_loss, noise_prob

    def label2multi_label(self, sentence_label):
        # transfer the sentence label from single label to multi label
        multi_sentence_label = torch.rand(
            (len(sentence_label), self.args.num_class)
        ).cuda()
        multi_sentence_label = multi_sentence_label * 0.2
        fill_num = torch.ones(multi_sentence_label.size()).cuda()
        fill_num = fill_num - multi_sentence_label
        multi_sentence_label.scatter_(
            1, sentence_label.unsqueeze(-1), fill_num
        ).long()
        # multi_sentence_label = (torch.rand(
        #     (len(sentence_label), self.args.num_class)
        # ) > 0.5).cuda().float()
        return multi_sentence_label

    def predict(self, batch):
        inputs, label = self.batch_generate(batch)

        # forward
        self.model.eval()
        inputs, sentence_label = self.batch_divide(inputs)
        logits, sentence_logits, noise_logits = \
            self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction="mean")
        acc = self.acc_sum(logits, label)
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()

        sentence_acc = self.acc_sum(sentence_logits, sentence_label)
        noise_prob = F.softmax(logits, dim=-1).data.cpu().numpy().tolist()
        return (
            loss.data,
            acc,
            sentence_acc,
            predictions,
            label.data.cpu().numpy().tolist(),
            sentence_label.data.cpu().numpy().tolist(),
            predprob,
            noise_prob,
        )

    def acc_sum(self, logits, label):
        corrects = (
            torch.max(logits, 1)[1].view(label.size()).data == label.data
        ).sum()
        acc = 100.0 * np.float(corrects)
        return acc

    def evaluate(self, data_loader):
        return self.evaluate_for_pseudo(data_loader)

    def evaluate_for_pseudo(self, data_loader):
        predictions, labels, noise_probs = [], [], []
        val_loss, val_acc, val_step = 0.0, 0.0, 0
        val_sent_acc, val_sentence_step = 0.0, 0
        for i, batch in enumerate(data_loader):
            loss, acc, sent_acc, pred, label, sent_label, _, noise_prob \
                = self.predict(batch)
            val_loss += loss
            val_acc += acc
            val_sent_acc += sent_acc
            predictions += pred
            labels += label
            noise_probs += noise_prob
            val_step += len(label)
            val_sentence_step += len(sent_label)
        # f1 score
        f1_score = metrics.f1_score(labels, predictions, average="macro")
        return (
            val_loss / val_step,
            val_acc / val_step,
            f1_score,
            val_sent_acc / val_sentence_step,
            np.array(noise_probs).mean(axis=0).tolist()
        )

    def batch_divide(self, batch):
        '''
        return input, label
        '''
        return batch[0:-1], batch[-1]

    def batch_generate(self, batch):
        # convert to cuda
        batch = [b.cuda() for b in batch]
        # unpack inputs and label
        inputs, label = self.batch_divide(batch)
        return inputs, label

    def level_change(self, epoch, i):
        # print(epoch, i, self.step)
        ret = ''
        if epoch < self.args.begin_epoch:
            ret = 'aspect'
        else:
            if i % self.step == epoch % self.step:
                ret = 'sentence'
            else:
                ret = 'aspect'
        return ret


class Result_Record():
    def __init__(self, name):
        self.aspect_acc = 0.0
        self.aspect_loss = 0.0
        self.sentence_acc = 0.0
        self.aspect_step = 0
        self.sentence_step = 0
        self.name = name

    def update(
        self, loss, acc,
        G_loss_total, D_loss_total,
        noise_prob_total
    ):
        if type(loss) == tuple or type(acc) == tuple:
            self.sentence_step += 1
        if type(loss) == tuple and len(loss) == 4:
            (G_loss, D_loss, loss, noise_prob) = loss
            G_loss_total += G_loss
            D_loss_total += D_loss
            noise_prob_total += noise_prob

        if type(acc) == tuple and len(acc) == 2:
            (acc, sentence_acc) = acc
            self.sentence_acc += sentence_acc
        self.aspect_loss += loss
        self.aspect_acc += acc
        self.aspect_step += 1

        return (
            G_loss_total,
            D_loss_total,
            noise_prob_total,
        )

    def print_aspect(self):
        print(
            "aspect_loss: {:.6f}, aspect_acc: {:.6f}".format(
                self.aspect_loss / self.aspect_step,
                self.aspect_acc / self.aspect_step
            ),
            end=' '
        )
        self.aspect_acc = 0.0
        self.aspect_loss = 0.0
        self.aspect_step = 0

    def print_sentence(self):
        print(
            "sentence_acc: {:.6f}".format(
                self.sentence_acc / self.sentence_step
            ),
            end=' '
        )
        self.sentence_acc = 0.0
        self.sentence_step = 0
