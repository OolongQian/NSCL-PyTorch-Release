#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_nscl_derender.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/10/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Derendering model for the Neuro-Symbolic Concept Learner.

Unlike the model in NS-VQA, the model receives only ground-truth programs and needs to execute the program
to get the supervision for the VSE modules. This model tests the implementation of the differentiable
(or the so-called quasi-symbolic) reasoning process.

Note that, in order to train this model, one must use the curriculum learning.
"""

"""qian: after building dataset, we dive into learning model. 
    the model takes the role of, a reasoning agent, and a differentiable symbolic inference process.
    hence, we look at it in both a philosophical and a technical point of views."""

from jacinle.utils.container import GView
from nscl.models.reasoning_v1 import make_reasoning_v1_configs, ReasoningV1Model
from nscl.models.utils import canonize_monitors, update_from_loss_module

configs = make_reasoning_v1_configs()
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True


class Model(ReasoningV1Model):
    """pass"""
    """qian: This model is specific for NSCL CLEVR curriculum experiment,  
        and it inherits ReasoningV1Model. 
        I would like to check out the base ReasoningV1Model first."""
    def __init__(self, vocab):
        super().__init__(vocab, configs)

    def forward(self, feed_dict):
        # qian: GView seems a fancy custom version of python dict.
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        """qian: checkout the input and output shape and semantics of these intermediate 
            results along the pipeline."""

        # qian: feed_dict.image (32, 3, 256, 384)
        # f_scene.shape (32, 256, 16, 24)
        f_scene = self.resnet(feed_dict.image)
        # qian: feed_dict.objects (96, 4)
        # feed_dict.objects_length (32)
        f_sng = self.scene_graph(f_scene, feed_dict.objects, feed_dict.objects_length)

        programs = feed_dict.program_qsseq
        programs, buffers, answers = self.reasoning(f_sng, programs, fd=feed_dict)

        print("print program and answer pairs given by the reasoning module...")
        for prog, ans in zip(programs, answers):
            print(prog)
            print(ans)
            print()

        outputs['buffers'] = buffers
        outputs['answer'] = answers

        update_from_loss_module(monitors, outputs, self.scene_loss(
            feed_dict, f_sng,
            self.reasoning.embedding_attribute, self.reasoning.embedding_relation
        ))
        update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))

        canonize_monitors(monitors)
        print("check loss:", monitors.keys())
        print("check whether self.qa_loss add supervision", self.qa_loss.add_supervision)  # qian: True.
        print("print configs.train.scene_add_supervision", configs.train.scene_add_supervision)

        if self.training:
            loss = monitors['loss/qa']
            if configs.train.scene_add_supervision:  # qian: no scene supervision.
                loss = loss + monitors['loss/scene']
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            outputs['buffers'] = buffers
            return outputs


def make_model(args, vocab):
    return Model(vocab)
