import os
import pdb
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F
import tensorflow as tf

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0, beam_width=10):
        self.gloss_dict = gloss_dict
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.beam_width= beam_width
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=beam_width, blank_id=blank_id,
                                                    num_processes=10)
    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
 
        return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        # ret_list1= self.tf_BeamSearch(nn_output, vid_lgt, probs)
        # return ret_list1
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''

        index_list = torch.argmax(nn_output.cpu(), axis=2)
        batchsize, lgt = index_list.shape
        blank_rate =[]
        for batch_idx in range(batchsize):
            group_result = [x.item() for x in index_list[batch_idx][:int(vid_lgt[batch_idx])]]
            blank_rate.append(group_result)


        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            # ret_list.append([str(int(gloss_id)) for idx, gloss_id in enumerate(first_result)])
            ret_list.append([self.gloss_dict.convert_ids_to_tokens(int(gloss_id)) for idx, gloss_id in
                             enumerate(first_result)]+["<pad>"])
        # if ret_list!= ret_list1:
        #     print("debug==", ret_list)
        #     print("debu1==", ret_list1)
        return ret_list

  