# encoding utf-8
import argparse


def arg_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/Restaurants")
    parser.add_argument("--vocab_dir", type=str, default="dataset/Restaurants")
    parser.add_argument(
        "--hidden_dim", type=int, default=768, help="bert dim."
    )

    parser.add_argument(
        "--dep_dim", type=int, default=30, help="dep embedding dimension."
    )
    parser.add_argument(
        "--pos_dim", type=int, default=0, help="pos embedding dimension."
    )
    parser.add_argument(
        "--post_dim", type=int, default=0, help="position embedding dimension."
    )
    parser.add_argument(
        "--num_class", type=int, default=3, help="Num of sentiment class."
    )

    parser.add_argument(
        "--input_dropout", type=float, default=0.1, help="Input dropout rate."
    )
    parser.add_argument(
        "--layer_dropout", type=float, default=0,
        help="RGAT layer dropout rate."
    )
    parser.add_argument(
        "--att_dropout", type=float, default=0,
        help="self-attention layer dropout rate."
    )
    parser.add_argument("--lower", default=True, help="Lowercase all words.")
    parser.add_argument("--direct", default=False)
    parser.add_argument("--loop", default=True)
    parser.add_argument("--reset_pooling", default=False, action="store_true")
    parser.add_argument(
        "--en_lr", type=float, default=2e-5, help="learning rate."
    )
    parser.add_argument(
        "--de_lr", type=float, default=2e-5, help="learning rate for bert."
    )
    parser.add_argument(
        "--bert_lr", type=float, default=2e-5, help="learning rate for bert."
    )
    parser.add_argument(
        "--l2", type=float, default=1e-6, help="weight decay rate."
    )
    parser.add_argument(
        "--optim",
        choices=["sgd", "adagrad", "adam", "adamax"],
        default="adam",
        help="Optimizer: sgd, adagrad, adam or adamax.",
    )
    parser.add_argument(
        "--num_layer", type=int, default=3, help="Number of graph layers."
    )
    parser.add_argument(
        "--num_epoch", type=int, default=20,
        help="Number of total training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--log_step", type=int, default=16, help="Print log every k steps."
    )
    parser.add_argument(
        "--save_dir", type=str, default="./saved_models/res14",
        help="Root dir for saving models."
    )
    parser.add_argument(
        "--model", type=str, default="SGAT",
        help="model to use, (std, GAT, SGAT)"
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--bert_out_dim", type=int, default=100)
    parser.add_argument(
        "--output_merge",
        type=str,
        default="gatenorm2",
        help="merge method to use, (none, addnorm, add, attn, gate, gatenorm2)"
    )
    parser.add_argument("--max_len", type=int, default=80)
    ########
    parser.add_argument("--pre_train", type=int, default=0)
    parser.add_argument("--auto_lr", type=int, default=0)
    parser.add_argument("--sentence_weight", type=float, default=1)
    parser.add_argument("--anti_weight", type=float, default=1)
    parser.add_argument("--pseudo_weight", type=float, default=1)
    parser.add_argument("--step_long", type=int, default=2)
    parser.add_argument("--begin_epoch", type=int, default=0)
    parser.add_argument("--bert_for_sentence", type=int, default=0)
    parser.add_argument("--pseudo_label", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--aux_round", type=int, default=1)
    parser.add_argument("--dataset_size", type=str, default='10k')
    parser.add_argument("--bert_from", type=str, default='bert-base-uncased')
    parser.add_argument("--decay_patience", type=int, default=3)

    ########
    return parser
