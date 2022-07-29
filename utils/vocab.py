# encoding utf-8
# zym 2020.1.22

from vocabulary.vocab import Vocab


def vocab_generate(args=None):
    # load vocab
    print("Loading vocab...")
    # token
    token_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_tok.vocab")
    # deprel
    dep_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_dep.vocab")
    # polarity
    pol_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pol.vocab")
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
    return vocab
