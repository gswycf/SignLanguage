import torch,sys,math
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig, MBartModel
from modules.tokenizer import *
from utils.misc import freeze_params
from torch import nn, Tensor
from torch.autograd import Variable


def load_model(path):
    model = MBartForConditionalGeneration.from_pretrained(path, attention_dropout= 0.1, dropout= 0.3)
    return model


class SLTModel(nn.Module):
    def __init__(self, input_type="feature", cfg=None):
        super().__init__()
        self.input_type = input_type
        assert self.input_type in ['gloss','feature', 'feature+gloss']
        self.model = load_model(cfg['pretrained_model_name_or_path'])

        self.input_dim = self.model.config.d_model
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.d_model))
        #
        # self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg["GlossTokenizer"])
        # self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg["TextTokenizer"])

        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index,
            smoothing=cfg['label_smoothing'])


    def prepare_text_inputs(self, input_str):
        input_ids = self.text_tokenizer(input_str)
        return input_ids["labels"], input_ids["decoder_input_ids"]

    def prepare_feature_inputs(self, input_feature, input_lengths, gloss=None, gloss_lengths=None):
        suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index,:]]
        src_lang_id = self.text_tokenizer.lang_index 
        src_lang_code_embedding = self.model.model.shared.weight[src_lang_id,:]
        suffix_embedding.append(src_lang_code_embedding)
        suffix_len = len(suffix_embedding)
        suffix_embedding = torch.stack(suffix_embedding, dim=0)
        max_length = torch.max(input_lengths)+suffix_len
        if 'gloss' in self.input_type:
            gloss = self.text_tokenizer.getids(gloss)
            for ii, in_len in enumerate(input_lengths):
                max_length = max(max_length, in_len + len(gloss[ii])+suffix_len)


        inputs_embeds = []
        attention_mask = torch.zeros([input_feature.shape[0], max_length], dtype=torch.long, device=input_feature.device)

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii]
            valid_feature = feature[:valid_len,:] #t,D

            if 'gloss' in self.input_type:
                gloss_lengths = len(gloss[ii]) #input_lengths[ii]+
                gloss_emdb = torch.stack([self.model.model.shared.weight[glossids] for glossids in gloss[ii]]) #g_l D
                # print("debug1==", valid_feature.shape, gloss_emdb.shape)
                # valid_feature = torch.cat([valid_feature, gloss_emdb.detach()], dim=0) # t+g_l, D
                # valid_len = valid_len + gloss_lengths
                # print("debug111=",  valid_feature.shape)
                # print("debug22==", valid_feature.shape)
                valid_feature = torch.cat( [gloss_emdb, feature[:valid_len - gloss_lengths, :]], dim=0)
            if suffix_embedding != None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0) # t+2, D
            else:
                feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0]<max_length:
                pad_len = max_length-feature_w_suffix.shape[0]
                padding = torch.zeros([pad_len, feature_w_suffix.shape[1]],
                    dtype=feature_w_suffix.dtype, device=feature_w_suffix.device)
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0) #t+2+pl,D
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len+suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0)*self.input_embed_scale, #B,T,D
            'attention_mask': attention_mask#attention_mask
        }
        return transformer_inputs



    def prepare_gloss_inputs(self, gloss):

        suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index, :]]
        src_lang_id = self.text_tokenizer.lang_index  # self.text_tokenizer.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]

        src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
        suffix_embedding.append(src_lang_code_embedding)
        suffix_len = len(suffix_embedding)
        suffix_embedding = torch.stack(suffix_embedding, dim=0)

        gloss,attention_mask = self.text_tokenizer.getidsforgloss(gloss)
        return gloss.to(self.model.device), torch.tensor(attention_mask).to(self.model.device)
       
    #phoenix 14t 100
    def generate(self, gloss=None, num_beams=6, max_length=60, length_penalty=1,
                 input_feature=None, input_lengths=None):
        if self.input_type=='gloss':
            inputs_ids, attention_mask = self.prepare_gloss_inputs(gloss)
            assert attention_mask != None
            batch_size = attention_mask.shape[0]
            decoder_input_ids = torch.ones([batch_size, 1], dtype=torch.long,
                                           device=attention_mask.device) * self.text_tokenizer.sos_index
            assert inputs_ids != None and attention_mask != None
            output_dict = self.model.generate(
                input_ids=inputs_ids, attention_mask=attention_mask,  # same with forward
                decoder_input_ids=decoder_input_ids,
                num_beams=num_beams, length_penalty=length_penalty, max_length=max_length,
                return_dict_in_generate=True, output_hidden_states=True)
            output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
            # print("dddd==", output_dict.keys())
            return output_dict

        else:
            nkwargs = self.prepare_feature_inputs(input_feature, input_lengths, gloss)
            inputs_embeds, attention_mask = nkwargs['inputs_embeds'], nkwargs['attention_mask']
        assert attention_mask!=None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones([batch_size,1],dtype=torch.long, device=attention_mask.device)*self.text_tokenizer.sos_index
        assert inputs_embeds!=None and attention_mask!=None
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, #same with forward
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length,
            return_dict_in_generate=True, output_hidden_states=True)
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        # print("dddd==", output_dict.keys())
        return output_dict


