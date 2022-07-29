# encoding=utf-8
import torch
import numpy as np
from utils import arg
from vocabulary.vocab import Vocab
from utils import helper
from utils import seed
from utils.arg import arg_generate

from loader import ABSADataLoader
from train import ABSATrainer, Result_Record
from visual.curve import plot_train_curve


parser = arg_generate()
args = parser.parse_args()

# set random seed
seed.seed_init(args.seed)
helper.print_arguments(args)

# load vocab
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_tok.vocab")  # token
dep_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_dep.vocab")  # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pol.vocab")  # polarity
vocab = (token_vocab, dep_vocab, pol_vocab)
print(
    "token_vocab: {}, dep_vocab: {}, pol_vocab: {}"
    .format(
        len(token_vocab),
        len(dep_vocab),
        len(pol_vocab)
    )
)
args.tok_size = len(token_vocab)
args.dep_size = len(dep_vocab)

# load data
print("Loading data from {} with batch size {}...".format(
    args.data_dir, args.batch_size)
)

train_batch = ABSADataLoader(
    args.data_dir + "/" + args.dataset_size + "/train.json",
    args.batch_size, args, vocab, shuffle=True
)
valid_batch = ABSADataLoader(
    args.data_dir + "/" + args.dataset_size + "/test.json",
    args.batch_size, args, vocab, shuffle=False
)
test_batch = ABSADataLoader(
    args.data_dir + "/" + args.dataset_size + "/test.json",
    args.batch_size, args, vocab, shuffle=False
)

if args.debug == 1:
    aux_batch = ABSADataLoader(
        args.data_dir + "/" + args.dataset_size + "/test.json",
        args.batch_size, args, vocab, shuffle=True
    )
else:
    aux_batch = ABSADataLoader(
        args.data_dir + "/" + args.dataset_size + "/auxiliary.json",
        args.batch_size, args, vocab, shuffle=True
    )

# check saved_models director
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)


def _totally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


trainer = ABSATrainer(args)
# print(trainer.model)
print('# parameters:', _totally_parameters(trainer.model))
best_path = model_save_dir + "/best_model.pt"
print("Training Set: {}".format(len(train_batch)))
print("Valid/Test Set: {}".format(len(test_batch)))

train_acc_history, train_loss_history = [0.0], [0.0]
val_acc_history, val_loss_history, val_f1_score_history = \
    [0.0], [0.0], [0.0]
test_acc_history, test_loss_history, test_f1_score_history = \
    [0.0], [0.0], [0.0]

# for aux_batch to iter
key = 0
lr_decay = 0

for epoch in range(1, args.num_epoch + 1):
    print("Epoch {}".format(epoch) + "-" * 60)
    aspect_record = Result_Record('aspect')
    sent_record = Result_Record('sentence')
    G_D_step = 0
    G_loss_total, D_loss_total = 0.0, 0.0
    train_loss, train_acc, train_step = 0.0, 0.0, 0
    noise_prob_total = np.array([0.0, 0.0, 0.0])
    for i, batch in enumerate(train_batch):
        train_step += 1
        G_D_step += 1
        level = trainer.level_change(epoch, i)
        loss, acc = trainer.update(
            batch,
            level='aspect'
        )
        (
            G_loss_total, D_loss_total,
            noise_prob_total
        ) = aspect_record.update(
            loss, acc,
            G_loss_total, D_loss_total,
            noise_prob_total
        )

        for _ in range(args.aux_round):
            sentence_batch = aux_batch.__getitem__(key % len(aux_batch))
            key += 1
            loss, acc = trainer.update(
                sentence_batch,
                level='sentence'
            )

            (
                G_loss_total, D_loss_total,
                noise_prob_total
            ) = sent_record.update(
                loss, acc,
                G_loss_total, D_loss_total,
                noise_prob_total
            )

        # evaluate
        if train_step % args.log_step == 0:
            train_loss += aspect_record.aspect_loss
            train_acc += aspect_record.aspect_acc
            print("{}/{}".format(i, len(train_batch)), end=' ')
            aspect_record.print_aspect()
            print('Pseudo', end=' ')
            sent_record.print_aspect()
            print('')
            print("{}/{}".format(i, len(train_batch)), end=' ')
            sent_record.print_sentence()
            print('Pseudo', end=' ')
            aspect_record.print_sentence()
            rounds = args.aux_round + 1
            print(
                'noise prob:',
                (noise_prob_total / (G_D_step*rounds)).tolist(),
                end=' '
            )
            print('G_loss: {:.6f}, D_loss: {:.6f}'.format(
                G_loss_total / (G_D_step*rounds),
                D_loss_total / (G_D_step*rounds),
            ))
            G_D_step = 0
            G_loss_total, D_loss_total = 0.0, 0.0
            noise_prob_total = np.array([0.0, 0.0, 0.0])

    val_loss, val_acc, val_f1, sent_acc, noise_p = \
        trainer.evaluate(valid_batch)

    train_loss += aspect_record.aspect_loss
    train_acc += aspect_record.aspect_acc

    print(
        "End of {} train_loss: {:.4f}, train_acc: {:.4f},"
        .format(
            epoch, train_loss / train_step, train_acc / train_step
        ), end=' '
    )
    print(
        "val_loss: {:.4f}, val_acc: {:.4f}, f1_score: {:.4f}"
        .format(val_loss, val_acc, val_f1)
    )
    print("sentence acc: {:.4f}".format(sent_acc))
    print("noise distribution:", noise_p)

    train_acc_history.append(sent_acc)
    val_loss_history.append(val_loss)

    # save best model
    if epoch == 1 or float(val_acc) > max(val_acc_history):
        torch.save(trainer, best_path)
        print("new best model saved.")

    # change learning rate
    if bool(args.auto_lr) is True:
        if epoch > args.decay_patience and lr_decay > args.decay_patience and \
                float(val_acc) < max(val_acc_history):
            trainer.lr_update()
            lr_decay = 0
        else:
            lr_decay += 1

    val_acc_history.append(float(val_acc))
    val_f1_score_history.append(val_f1)


loss_picture = model_save_dir + '/loss'
plot_train_curve(
    'epoch', 'loss', args.num_epoch,
    val_loss_history[1:],
    img_path=loss_picture
)
acc_picture = model_save_dir + '/acc'
plot_train_curve(
    'epoch', 'acc', args.num_epoch,
    val_acc_history[1:], train_acc_history[1:],
    img_path=acc_picture
)


print("Training ended with {} epochs.".format(epoch))
print("the first of the best acc appeared at {} epoch".format(
    val_acc_history.index(max(val_acc_history))
))
print("Loading best checkpoint from ", best_path)
trainer = torch.load(best_path)
test_loss, test_acc, test_f1, sent_acc, noise_p = trainer.evaluate(test_batch)
print(
    "Evaluation Results: test_loss:{}, test_acc:{}, test_f1:{}".format(
        test_loss, test_acc, test_f1
    )
)
print("sentence acc: {:.4f}".format(sent_acc))
print("noise distribution:", noise_p)
