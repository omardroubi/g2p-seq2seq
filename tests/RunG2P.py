import sys
import os
from g2p_seq2seq import g2p
import g2p_seq2seq.g2p_trainer_utils as g2p_trainer_utils
from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params

print("hello")
model_dir = os.path.abspath("models/g2p-seq2seq")
gt_path = os.path.abspath("data/toydict.test")
params = Params(model_dir, gt_path)
params.hparams = g2p_trainer_utils.load_params(model_dir)
g2p_model = G2PModel(params, test_path=gt_path)
g2p_model.decode_word("hello")