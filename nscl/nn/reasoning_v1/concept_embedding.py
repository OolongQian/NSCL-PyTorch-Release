#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : concept_embedding.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
We consider three types of
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch
import jactorch.nn as jacnn

from jacinle.utils.cache import cached_property

__all__ = [
    'AttributeBlock', 'ConceptBlock', 'ConceptEmbedding'
]


"""QAS mode: if True, when judging if two objects are of same color, consider all concepts belongs to `color`."""
_query_assisted_same = False


def set_query_assisted_same(value):
    """Set the QAS mode."""
    global _query_assisted_same
    _query_assisted_same = value


class AttributeBlock(nn.Module):
    """Attribute as a neural operator."""
    """Attribute is a category while concepts belong to one or more categories.
        AttributeBlock is implemented as a linear layer.
        Example:
            identifier: 'color'.
            input_dim, output_dim: 256, 64.
        Here, the identifier is managed by ConceptEmbedding class."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.map = jacnn.LinearLayer(input_dim, output_dim, activation=None)


class ConceptBlock(nn.Module):
    """
    Concept is an embedding in the corresponding attribute space.
    For example,
        an concept is 'gray', and thus a conceptBlock is created to represent this concept in latent space.
        The key 'gray' is managed by ConceptEmbedding class.
        The exact feature value of 'gray' is left to torch.randn().
    """
    def __init__(self, embedding_dim, nr_attributes, attribute_agnostic=False):
        """

        Args:
            embedding_dim (int): dimension of the embedding.
            nr_attributes (int): number of known attributes.
            attribute_agnostic (bool): if the embedding in different embedding spaces are shared or not.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.nr_attributes = nr_attributes
        self.attribute_agnostic = attribute_agnostic

        if self.attribute_agnostic:
            self.embedding = nn.Parameter(torch.randn(embedding_dim))
        else:
            self.embedding = nn.Parameter(torch.randn(nr_attributes, embedding_dim))
        self.belong = nn.Parameter(torch.randn(nr_attributes) * 0.1)

        self.known_belong = False

    def set_belong(self, belong_id):
        """
        Set the attribute that this concept belongs to.
        qian: It may be because, 'belong' relates to the PyTorch control flow,
            therefore it should be PyTorch Tensor.
            And we set .requires_grad to False, to make it a static configuration kind of thing.

        Args:
            belong_id (int): the id of the attribute.
        """
        self.belong.data.fill_(-100)
        self.belong.data[belong_id] = 100
        self.belong.requires_grad = False
        self.known_belong = True

    @property
    def normalized_embedding(self):
        """L2-normalized embedding in all spaces."""
        embedding = self.embedding / self.embedding.norm(2, dim=-1, keepdim=True)
        if self.attribute_agnostic:
            return jactorch.broadcast(embedding.unsqueeze(0), 0, self.nr_attributes)
        return embedding

    @property
    def log_normalized_belong(self):
        """Log-softmax-normalized belong vector."""
        return F.log_softmax(self.belong, dim=-1)

    @property
    def normalized_belong(self):
        """Softmax-normalized belong vector."""
        return F.softmax(self.belong, dim=-1)


class ConceptEmbedding(nn.Module):
    """qian: Concept embedding is a container of a bunch of attribute categories and concepts.
        It is a wrapper over the ontology used."""
    def __init__(self, attribute_agnostic):
        super().__init__()

        self.attribute_agnostic = attribute_agnostic
        self.all_attributes = list()
        self.all_concepts = list()
        self.attribute_operators = nn.Module()
        self.concept_embeddings = nn.Module()

    @property
    def nr_attributes(self):
        return len(self.all_attributes)

    @property
    def nr_concepts(self):
        return len(self.all_concepts)

    @cached_property
    def attribute2id(self):
        return {a: i for i, a in enumerate(self.all_attributes)}

    def init_attribute(self, identifier, input_dim, output_dim):
        """Attribute is a category while concepts belong to one or more categories.
            AttributeBlock is implemented as a linear layer.
            Example:
                identifier: 'color'.
                input_dim, output_dim: 256, 64."""
        assert self.nr_concepts == 0, 'Can not register attributes after having registered any concepts.'
        self.attribute_operators.add_module('attribute_' + identifier, AttributeBlock(input_dim, output_dim))
        self.all_attributes.append(identifier)
        # TODO(Jiayuan Mao @ 11/08): remove this sorting...
        self.all_attributes.sort()

    def init_concept(self, identifier, input_dim, known_belong=None):
        """Concept is a feature embedding with a specific identifier.
            Each concept should belong to a corresponding attribute,
                which is not initialized during creation, but set by 'set_belong'.
            Example:
                identifier: gray."""
        block = ConceptBlock(input_dim, self.nr_attributes, attribute_agnostic=self.attribute_agnostic)
        """qian: add submodule to the outer module. 
            The added submodule can be accessed by dot with name, 
            as well as getattr()."""
        self.concept_embeddings.add_module('concept_' + identifier, block)
        if known_belong is not None:
            block.set_belong(self.attribute2id[known_belong])
        self.all_concepts.append(identifier)

    def get_belongs(self):
        belongs = dict()
        for k, v in self.concept_embeddings.named_children():
            belongs[k] = self.all_attributes[v.belong.argmax(-1).item()]
        class_based = dict()
        for k, v in belongs.items():
            class_based.setdefault(v, list()).append(k)
        class_based = {k: sorted(v) for k, v in class_based.items()}
        return class_based

    def get_attribute(self, identifier):
        x = getattr(self.attribute_operators, 'attribute_' + identifier)
        return x.map

    def get_all_attributes(self):
        return [self.get_attribute(a) for a in self.all_attributes]

    def get_concept(self, identifier):
        return getattr(self.concept_embeddings, 'concept_' + identifier)

    def get_all_concepts(self):
        return {c: self.get_concept(c) for c in self.all_concepts}

    def get_concepts_by_attribute(self, identifier):
        return self.get_attribute(identifier), self.get_all_concepts(), self.attribute2id[identifier]

    _margin = 0.85
    _margin_cross = 0.5
    _tau = 0.25

    def similarity(self, query, identifier):
        mappings = self.get_all_attributes()
        concept = self.get_concept(identifier)

        # shape: [batch, attributes, channel] or [attributes, channel]
        query_mapped = torch.stack([m(query) for m in mappings], dim=-2)
        query_mapped = query_mapped / query_mapped.norm(2, dim=-1, keepdim=True)
        reference = jactorch.add_dim_as_except(concept.normalized_embedding, query_mapped, -2, -1)

        margin = self._margin
        logits = ((query_mapped * reference).sum(dim=-1) - 1 + margin) / margin / self._tau

        belong = jactorch.add_dim_as_except(concept.log_normalized_belong, logits, -1)
        logits = jactorch.logsumexp(logits + belong, dim=-1)

        return logits

    def similarity2(self, q1, q2, identifier, _normalized=False):
        """
        Args:
            _normalized (bool): backdoor for function `cross_similarity`.
        """

        global _query_assisted_same

        logits_and = lambda x, y: torch.min(x, y)
        logits_or = lambda x, y: torch.max(x, y)

        if not _normalized:
            q1 = q1 / q1.norm(2, dim=-1, keepdim=True)
            q2 = q2 / q2.norm(2, dim=-1, keepdim=True)

        if not _query_assisted_same or not self.training:
            margin = self._margin_cross
            logits = ((q1 * q2).sum(dim=-1) - 1 + margin) / margin / self._tau
            return logits
        else:
            margin = self._margin_cross
            logits1 = ((q1 * q2).sum(dim=-1) - 1 + margin) / margin / self._tau

            _, concepts, attr_id = self.get_concepts_by_attribute(identifier)
            masks = []
            for k, v in concepts:
                embedding = v.normalized_embedding[attr_id]
                embedding = jactorch.add_dim_as_except(embedding, q1, -1)

                margin = self._margin
                mask1 = ((q1 * embedding).sum(dim=-1) - 1 + margin) / margin / self._tau
                mask2 = ((q2 * embedding).sum(dim=-1) - 1 + margin) / margin / self._tau

                belong_score = v.normalized_belong[attr_id]
                # TODO(Jiayuan Mao @ 08/10): this line may have numerical issue.
                mask = logits_or(
                    logits_and(mask1, mask2),
                    logits_and(-mask1, -mask2),
                ) * belong_score

                masks.append(mask)
            logits2 = torch.stack(masks, dim=-1).sum(dim=-1)

            # TODO(Jiayuan Mao @ 08/09): should we take the average here? or just use the logits2?
            return torch.min(logits1, logits2)

    def cross_similarity(self, query, identifier):
        mapping = self.get_attribute(identifier)
        query = mapping(query)
        query = query / query.norm(2, dim=-1, keepdim=True)
        q1, q2 = jactorch.meshgrid(query, dim=-2)

        return self.similarity2(q1, q2, identifier, _normalized=True)

    def map_attribute(self, query, identifier):
        mapping = self.get_attribute(identifier)
        return mapping(query)

    def query_attribute(self, query, identifier):
        mapping, concepts, attr_id = self.get_concepts_by_attribute(identifier)
        query = mapping(query)
        query = query / query.norm(2, dim=-1, keepdim=True)

        word2idx = {}
        masks = []
        for k, v in concepts.items():
            embedding = v.normalized_embedding[attr_id]
            embedding = jactorch.add_dim_as_except(embedding, query, -1)

            margin = self._margin
            mask = ((query * embedding).sum(dim=-1) - 1 + margin) / margin / self._tau

            belong_score = v.log_normalized_belong[attr_id]
            mask = mask + belong_score

            masks.append(mask)
            word2idx[k] = len(word2idx)

        masks = torch.stack(masks, dim=-1)
        return masks, word2idx

