#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : datasets.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.


"""qian: this datasets.py inherits common.filterable.py's base dataset class definition.
    and implements the above functionality specific to NSCL.

    here, we dive deep into the data itself. i need to know something about data.
    now, i am more interested in understanding his entire architecture,
    thus i first checkout what he's doing on the high level,
    thus only understanding the outputs of the data-processing functions is OKay. """

import os.path as osp

import nltk
import numpy as np
from PIL import Image

import jacinle.io as io
from jacinle.logging import get_logger
from jacinle.utils.container import GView

from nscl.datasets.definition import gdef
from nscl.datasets.common.filterable import FilterableDatasetUnwrapped, FilterableDatasetView
from nscl.datasets.common.vocab import Vocab
from nscl.datasets.common.program_translator import nsclseq_to_nscltree, nsclseq_to_nsclqsseq, nscltree_to_nsclqstree, \
    gen_vocab

logger = get_logger(__file__)

__all__ = ['NSCLDataset', 'ConceptRetrievalDataset', 'ConceptQuantizationDataset']


class NSCLDatasetUnwrapped(FilterableDatasetUnwrapped):
    """pass"""
    """qian: filterableDataset is like a small relational-database, 
        here, we instantiate this form to clevr dataset and NSCL task. """
    def __init__(self, scenes_json, questions_json, image_root, image_transform, vocab_json, question_transform=None,
                 incl_scene=True):
        """pass"""
        """This initialization assigns file paths and reads data. 
            The read data variables are 'question', 'scenes', 'vocab'. 
            I record the data format here: 
                'vocab': idx2word, word2idx. 
                'questions':  
                    a list of question, 
                        'image_index', 
                        'question_index', 
                        'image_filename', 
                        'question_family_index', 
                        'split', 
                        'answer', 
                        'question': natural language question.
                'scenes': 
                    a list of scene, 
                        'image_index', 
                        'objects':
                            'color', 
                            'size', 
                            'rotation', 
                            'shape', 
                            '3d_coords', 
                            'material', 
                            'pixel_coords'
                        'relationships': i have not understood the semantics yet,  
                            'right', 
                            'behind', 
                            'front', 
                            'left', 
                        'image_filename', 
                        'split', 
                        'directions': i have not understood the semantics yet, 
                        'object_detection'.
                        """
        super().__init__()

        """qian: record some data file paths."""
        self.scenes_json = scenes_json
        self.questions_json = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.vocab_json = vocab_json
        self.question_transform = question_transform

        self.incl_scene = incl_scene

        """qian: load data."""
        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']
        if isinstance(self.questions_json, (tuple, list)):
            self.questions = list()
            for filename in self.questions_json:
                logger.info('Loading questions from: "{}".'.format(filename))
                self.questions.extend(io.load_json(filename)['questions'])
        else:
            logger.info('Loading questions from: "{}".'.format(self.questions_json))
            self.questions = io.load_json(self.questions_json)['questions']

        if self.vocab_json is not None:
            logger.info('Loading vocab from: "{}".'.format(self.vocab_json))
            self.vocab = Vocab.from_json(self.vocab_json)
        else:
            logger.info('Building the vocab.')
            self.vocab = gen_vocab(self)

    def _get_metainfo(self, index):
        """"""
        """qian: here, gdef is set to be NSCLDefinition class.
            Here, dataset's variables are set by Definition's methods. 
            Thus is the inter-play of two classes.
            
            This _get_metainfo retrieves the index-th data entry. 
            Put everything in 'question' variable and return it."""

        """qian: perception and question section, these are usual common data."""
        question = gdef.translate_question(self.questions[index])  # qian: here, actually do nothing.
        scene = gdef.translate_scene(self.scenes[question['image_index']])  # qian: here, actually do nothing.
        question['scene'] = scene

        question['image_index'] = question['image_index']
        question['image_filename'] = gdef.get_image_filename(scene)
        question['question_index'] = index
        question['question_tokenized'] = nltk.word_tokenize(question['question'])

        # program section
        has_program = False
        if 'program_nsclseq' in question:
            question['program_raw'] = question['program_nsclseq']
            question['program_seq'] = question['program_nsclseq']
            has_program = True
        elif 'program' in question:
            question['program_raw'] = question['program']
            # qian: note the 2nd argument is useless
            question['program_seq'] = gdef.program_to_nsclseq(question['program'], question)
            has_program = True

        """These program variations are cool. Let us check it out."""
        if has_program:
            # qian: program_tree converts list to tree.
            #   i mean, the input's reference is embedded in each operation.
            question['program_tree'] = nsclseq_to_nscltree(question['program_seq'])
            # qs- means with additional NSCL information.
            #   concept indices and values are attached.
            question['program_qsseq'] = nsclseq_to_nsclqsseq(question['program_seq'])
            question['program_qstree'] = nscltree_to_nsclqstree(question['program_tree'])
            question['question_type'] = question['program_seq'][-1]['op']
        else:
            question['question_type'] = None

        return question

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # metainfo annotations
        if self.incl_scene:
            feed_dict.scene = metainfo.scene
            feed_dict.update(gdef.annotate_objects(metainfo.scene))
            if 'objects' in feed_dict:
                # NB(Jiayuan Mao): in some datasets, object information might be completely unavailable.
                feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(gdef.annotate_scene(metainfo.scene))

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None and feed_dict.image_filename is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image, feed_dict.objects = self.image_transform(feed_dict.image, feed_dict.objects)

        # program
        if 'program_raw' in metainfo:
            feed_dict.program_raw = metainfo.program_raw
            feed_dict.program_seq = metainfo.program_seq
            feed_dict.program_tree = metainfo.program_tree
            feed_dict.program_qsseq = metainfo.program_qsseq
            feed_dict.program_qstree = metainfo.program_qstree
        feed_dict.question_type = metainfo.question_type

        # question
        feed_dict.question_index = metainfo.question_index
        feed_dict.question_raw = metainfo.question
        feed_dict.question_raw_tokenized = metainfo.question_tokenized
        feed_dict.question_metainfo = gdef.annotate_question_metainfo(metainfo)
        feed_dict.question = metainfo.question_tokenized
        feed_dict.answer = gdef.canonize_answer(metainfo.answer, metainfo.question_type)
        feed_dict.update(gdef.annotate_question(metainfo))

        if self.question_transform is not None:
            self.question_transform(feed_dict)
        feed_dict.question = np.array(self.vocab.map_sequence(feed_dict.question), dtype='int64')

        return feed_dict.raw()

    def __len__(self):
        return len(self.questions)


class NSCLDatsetFilterableView(FilterableDatasetView):
    """pass"""
    """create filterable nscl dataset.
        first three member functions are some filters defined for curriculum learning or something, 
        and the last is making dataloader out of this dataset. """
    def filter_program_size_raw(self, max_length):
        def filt(question):
            return len(question['program']) <= max_length

        return self.filter(filt, 'filter-program-size-clevr[{}]'.format(max_length))

    def filter_scene_size(self, max_scene_size):
        def filt(question):
            return len(question['scene']['objects']) <= max_scene_size

        return self.filter(filt, 'filter-scene-size[{}]'.format(max_scene_size))

    def filter_question_type(self, *, allowed=None, disallowed=None):
        def filt(question):
            if allowed is not None:
                return question['question_type'] in allowed
            elif disallowed is not None:
                return question['question_type'] not in disallowed

        if allowed is not None:
            return self.filter(filt, 'filter-question-type[allowed={{{{}}}]'.format(','.join(list(allowed))))
        elif disallowed is not None:
            return self.filter(filt, 'filter-question-type[disallowed={{{}}}]'.format(','.join(list(disallowed))))
        else:
            raise ValueError('Must provide either allowed={...} or disallowed={...}.')

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        """pass"""
        """This function creates dataloader out of our nscl dataset. 
            The use of pytorch dataloader is helpful, and i should learn it in detail. 
            I should learn python and pyTorch techniques via this opportunity."""
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'scene': 'skip',
            'objects_raw': 'skip',
            'objects': 'concat',

            'image_index': 'skip',
            'image_filename': 'skip',
            'question_index': 'skip',
            'question_metainfo': 'skip',

            'question_raw': 'skip',
            'question_raw_tokenized': 'skip',
            'question': 'pad',

            'program_raw': 'skip',
            'program_seq': 'skip',
            'program_tree': 'skip',
            'program_qsseq': 'skip',
            'program_qstree': 'skip',

            'question_type': 'skip',
            'answer': 'skip',
        }

        """qian: collate_guide defines how the features are processed when 
            encountering different situations."""
        gdef.update_collate_guide(collate_guide)

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


def NSCLDataset(*args, **kwargs):
    """this creates dataset for neural symbolic concept learner.
        the inner part should be its dataset instance, and the outer
        part should be a dataset wrapper. """
    return NSCLDatsetFilterableView(NSCLDatasetUnwrapped(*args, **kwargs))


class ConceptRetrievalDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, program, scenes_json, image_root, image_transform, incl_scene=True):
        super().__init__()

        self.program = program
        self.scenes_json = scenes_json
        self.image_root = image_root
        self.image_transform = image_transform

        self.incl_scene = incl_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']

    def _get_metainfo(self, index):
        question = dict()
        question['scene'] = gdef.translate_scene(self.scenes[index])

        question['image_index'] = index
        question['image_filename'] = gdef.get_image_filename(question['scene'])

        # program section
        question['program_raw'] = self.program
        question['program_seq'] = self.program
        question['program_tree'] = nsclseq_to_nscltree(question['program_seq'])
        question['program_qsseq'] = nsclseq_to_nsclqsseq(question['program_seq'])
        question['program_qstree'] = nscltree_to_nsclqstree(question['program_tree'])
        question['question_type'] = question['program_seq'][-1]['op']

        return question

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # scene annotations
        if self.incl_scene:
            feed_dict.scene = metainfo.scene
            feed_dict.update(gdef.annotate_objects(metainfo.scene))
            feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(gdef.annotate_scene(metainfo.scene))

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image, feed_dict.objects = self.image_transform(feed_dict.image, feed_dict.objects)

        # program
        feed_dict.program_raw = metainfo.program_raw
        feed_dict.program_seq = metainfo.program_seq
        feed_dict.program_tree = metainfo.program_tree
        feed_dict.program_qsseq = metainfo.program_qsseq
        feed_dict.program_qstree = metainfo.program_qstree
        feed_dict.question_type = metainfo.question_type

        # question
        feed_dict.answer = True

        return feed_dict.raw()

    def __len__(self):
        return len(self.scenes)


class ConceptRetrievalDatsetFilterableView(FilterableDatasetView):
    def filter_scene_size(self, max_scene_size):
        def filt(question):
            return len(question['scene']['objects']) <= max_scene_size

        return self.filter(filt, 'filter-scene-size[{}]'.format(max_scene_size))

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'scene': 'skip',
            'objects_raw': 'skip',
            'objects': 'concat',

            'image_index': 'skip',
            'image_filename': 'skip',

            'program_raw': 'skip',
            'program_seq': 'skip',
            'program_tree': 'skip',
            'program_qsseq': 'skip',
            'program_qstree': 'skip',

            'question_type': 'skip',
            'answer': 'skip',
        }

        gdef.update_collate_guide(collate_guide)

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


def ConceptRetrievalDataset(*args, **kwargs):
    return ConceptRetrievalDatsetFilterableView(ConceptRetrievalDatasetUnwrapped(*args, **kwargs))


class ConceptQuantizationDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, scenes_json, image_root, image_transform, incl_scene=True):
        super().__init__()

        self.scenes_json = scenes_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.incl_scene = incl_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']

    def _get_metainfo(self, index):
        question = dict()
        question['scene'] = gdef.translate_scene(self.scenes[index])

        question['image_index'] = index
        question['image_filename'] = gdef.get_image_filename(question['scene'])

        return question

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # scene annotations
        if self.incl_scene:
            feed_dict.scene = metainfo.scene
            feed_dict.update(gdef.annotate_objects(metainfo.scene))
            feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(gdef.annotate_scene(metainfo.scene))

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image, feed_dict.objects = self.image_transform(feed_dict.image, feed_dict.objects)

        return feed_dict.raw()

    def __len__(self):
        return len(self.scenes)


class ConceptQuantizationDatasetFilterableView(FilterableDatasetView):
    def filter_scene_size(self, max_scene_size):
        def filt(question):
            return len(question['scene']['objects']) <= max_scene_size

        return self.filter(filt, 'filter-scene-size[{}]'.format(max_scene_size))

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'scene': 'skip',
            'objects_raw': 'skip',
            'objects': 'concat',

            'image_index': 'skip',
            'image_filename': 'skip',
        }

        gdef.update_collate_guide(collate_guide)

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


def ConceptQuantizationDataset(*args, **kwargs):
    return ConceptQuantizationDatasetFilterableView(ConceptQuantizationDatasetUnwrapped(*args, **kwargs))
