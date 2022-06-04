import os
import spacy
import yaml
import numpy as np
from nltk.corpus import wordnet as wn





def load_config(config_file):
    with open("config/{}".format(config_file), "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    return config


class Parser:
    def match_engine(self, current_transformation):
        engines = self.config['engine']
        matched_engine = None
        current_transformation_str = ' '.join(current_transformation)

        for engine_name, engine in engines.items():
            for support_transformation in engine['support_transformations']:
                if support_transformation == current_transformation_str:
                    matched_engine = engine_name
                    break

            if matched_engine:
                break

        return matched_engine

    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.ontology_list = load_config("ontology.yaml")
        self.config = load_config("configuration.yaml")

    def extract_dependency(self, sentence):
        doc = self.nlp(sentence)

        pos_dict = {"NOUN": [], "ROOT": None}
        # token.head, token.children: dependencies

        for chunk in doc.noun_chunks:
            # print(chunk.text, chunk.root.text, chunk.root.dep_)
            if chunk.root.pos_ == "NOUN" and chunk.root.dep_ != "ROOT":
                pos_dict["NOUN"].append((chunk.root, chunk))

        for token in doc:
            # print(token.text, token.pos_, token.dep_)
            # if token.pos_ == "NOUN" and token.dep_ != "ROOT":
            #     pos_dict["NOUN"].append(token)
            if token.dep_ == "ROOT":
                pos_dict["ROOT"] = token

        return pos_dict

    def get_vector(self, word):
        token = self.nlp(word)
        return token.vector

    def get_similarity(self, word1, word2):
        t1 = self.nlp(word1)
        t2 = self.nlp(word2)
        distance = t1.similarity(t2)
        # v1 = w2v(model, word1)
        # v2 = w2v(model, word2)
        # distance = np.linalg.norm(v1-v2)
        return distance

    def get_wup_similarity(self, w1, w2, pos='noun'):
        if pos == 'verb':
            w1s = wn.synsets(w1, pos=wn.VERB)
            w2s = wn.synsets(w2, pos=wn.VERB)
        elif pos == 'noun':
            w1s = wn.synsets(w1, pos=wn.NOUN)
            w2s = wn.synsets(w2, pos=wn.NOUN)
        elif pos == 'adj':
            w1s = wn.synsets(w1, pos=wn.ADJ)
            w2s = wn.synsets(w2, pos=wn.ADJ)

        sims = []
        for w1 in w1s:
            for w2 in w2s:
                sims.append(w1.wup_similarity(w2))

        return max(sims)

    def parse_if(self, sentence, phase=1):
        pos_dict = self.extract_dependency(sentence)

        # identify ontology elements in the sentence
        ontology_element = {}
        for (noun, chunk) in pos_dict['NOUN']:
            word = noun.lemma_
            ontology_match = self.identify_ontology(word)
            if ontology_match:
                ontology_element[noun.text] = {'name': ontology_match, 'token': noun, 'chunk': chunk}

        ontology_element_update = {}
        for k, v in ontology_element.items():
            ontology_element_update[k] = v
            element = v['token']
            chunk = v['chunk']
            if type(self.ontology_list[v['name']]) == dict:
                for token in chunk:
                    # print(token.text)
                    if token.text != element.text:
                        for onto_property, property_vals in self.ontology_list[v['name']].items():
                            if token.text in property_vals:
                                ontology_element_update[k][onto_property] = token.text

                # predicates = {}
                #
                # for token in element.children:
                #     predicates[token.dep_] = (token, element)
                # print(predicates)
                # for onto_property, property_vals in self.ontology_list[v['name']].items():
                #     print(onto_property, property_vals)
                #     if k in property_vals:
                #         ontology_element_update[k][onto_property] = k
                #
                # for predicate_val in predicates.values():
                #     if predicate_val[0].text in property_vals:
                #         ontology_element_update[k][onto_property] = predicate_val[0].text

        # print(ontology_element_update)

        # identify transformation name
        root_verb = pos_dict['ROOT']

        transformation = self.identify_transformation(root_verb.lemma_)

        # identify transformation function
        subject = None
        predicates = {}
        doc = self.nlp(sentence)
        for token in doc:
            # for token in root_verb.children:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                predicates['nsubj'] = (token.text, token.head.text)
            elif token.dep_ in ['amod', 'advmod', 'dobj']:
                predicates[token.dep_] = (token.text, token.head.text)
            elif token.dep_ == 'prep':
                for t in token.children:
                    # if t.dep_ == 'pobj':
                    #     predicates[token.text] = (t.text, token.head.text)
                    # else:
                    predicates[token.dep_] = (token.text, t.text, token.head.text)

        transformation_predicate = None
        if transformation == 'add':
            if phase == 2 and 'closer' in sentence:
                transformation_predicate = ('add', 'closer')
            elif 'nsubj' in predicates.keys() and predicates['nsubj'][0] in ontology_element_update.keys() \
                    and predicates['nsubj'][0] not in self.config['environment'] \
                    and 'prep' in predicates.keys() and predicates['prep'][0] == 'on' and \
                    predicates['prep'][1] in ontology_element_update.keys():
                transformation_predicate = ('add', ontology_element_update[predicates['nsubj'][0]]['name'],
                                            ontology_element_update[predicates['prep'][1]]['name'], 'on')
            elif 'nsubj' in predicates.keys() and predicates['nsubj'][0] in ontology_element_update.keys() \
                    and 'prep' in predicates.keys() and predicates['prep'][0] == 'front' and \
                    predicates['prep'][1] in ontology_element_update.keys():
                transformation_predicate = ('add', ontology_element_update[predicates['nsubj'][0]]['name'],
                                            ontology_element_update[predicates['prep'][1]]['name'], 'front')
            elif 'nsubj' in predicates.keys() and predicates['nsubj'][0] in ontology_element_update.keys() \
                    and 'prep' in predicates.keys() and predicates['prep'][0] == 'behind' and \
                    predicates['prep'][1] in ontology_element_update.keys():
                transformation_predicate = ('add', ontology_element_update[predicates['nsubj'][0]]['name'],
                                            ontology_element_update[predicates['prep'][1]]['name'], 'behind')
            elif len(ontology_element_update) == 1 and list(ontology_element_update.keys())[0] in self.config[
                'environment']:
                transformation_predicate = (
                    'replace', ontology_element_update[list(ontology_element_update.keys())[0]]['name'], 'environment')
            elif 'nsubj' in predicates.keys() and predicates['nsubj'][0] in ontology_element_update.keys() and \
                    'prep' in predicates.keys() and predicates['prep'][1] in ontology_element_update.keys() and \
                    ontology_element_update[predicates['prep'][1]]['name'] in self.config['environment']:
                transformation_predicate = ('replace', ontology_element_update[predicates['nsubj'][0]]['name'],
                                            ontology_element_update[predicates['prep'][1]]['name'], 'environment')

        elif transformation == "remove":
            if 'nsubj' in predicates.keys() and predicates['nsubj'][0] in ontology_element_update.keys():
                transformation_predicate = ('remove', ontology_element_update[predicates['nsubj'][0]]['name'])

        elif transformation == "replace":
            if 'nsubj' in predicates.keys() and predicates['nsubj'][0] in ontology_element_update.keys() and \
                    'prep' in predicates.keys() and predicates['prep'][1] in ontology_element_update.keys() and \
                    ontology_element_update[predicates['prep'][1]]['name'] in self.config['object']:
                transformation_predicate = ('replace', ontology_element_update[predicates['nsubj'][0]]['name'],
                                            ontology_element_update[predicates['prep'][1]]['name'], 'object')
            elif 'nsubj' in predicates.keys() and predicates['nsubj'][0] in ontology_element_update.keys() and \
                    'prep' in predicates.keys() and predicates['prep'][1] in ontology_element_update.keys() and \
                    ontology_element_update[predicates['prep'][1]]['name'] in self.config['environment']:
                transformation_predicate = ('replace', ontology_element_update[predicates['nsubj'][0]]['name'],
                                            ontology_element_update[predicates['prep'][1]]['name'], 'environment')

        # print(transformation_predicate)
        return transformation_predicate

    def parse_then(self, sentence, phase=1):
        pos_dict = self.extract_dependency(sentence)
        change_action = self.identify_change_action(pos_dict['ROOT'].text)
        # print(change_action)
        change_subject = None
        change_type = None
        change_extent = None
        change_modifier = None
        change_comparative = None
        doc = self.nlp(sentence)

        if 'steer' in sentence:
            change_subject = 'steering'
        else:
            change_subject = 'speed'

        for ent in doc.ents:
            if ent.label_ == 'CARDINAL':
                change_type = 'number'
                change_extent = ent.text.split()[-1]
                if 'no less than' in ent.text:
                    change_modifier = 'more than'
                elif 'less than' in ent.text:
                    change_modifier = 'less than'
                elif 'no more than' in ent.text:
                    change_modifier = 'less than'
                elif 'more than' in ent.text:
                    change_modifier = 'more than'
                elif 'at least' in ent.text:
                    change_modifier = 'at least'
                break
            elif ent.label_ == 'PERCENT':
                change_type = 'percentage'
                change_extent = ent.text.split()[-1]
                if 'less than' in ent.text:
                    change_modifier = 'less than'
                elif 'more than' in ent.text:
                    change_modifier = 'more than'
                elif 'at least' in ent.text:
                    change_modifier = 'at least'

        if phase == 2:
            for token in doc:
                if token.tag_ == 'RBR' and token.text == 'more':
                    change_comparative = 'more'
                elif token.tag_ == 'RBR' and token.text == 'less':
                    change_comparative = 'less'

        return (change_subject, change_action, change_extent, change_type, change_modifier, change_comparative)

    def create_mr(self, expected_change, phase=1):
        mr = None
        (change_subject, change_action, change_extent, change_type, change_modifier,
         change_comparative) = expected_change
        if phase == 1 or (phase == 2 and change_comparative is None):
            if change_action == 'decrease' and change_modifier is None \
                and change_extent is None:
                mr = 'x{} - x1 < 0'.format(phase+1)
            elif change_action == 'increase' and change_modifier is None \
                and change_extent is None:
                mr = 'x{} - x1 > 0'.format(phase+1)
            elif change_action == 'decrease' and change_type == 'number' \
                    and (change_modifier == 'at least' or change_modifier == 'more than'):
                mr = 'x1 - x{} >= {}'.format(phase+1, change_extent)
            elif change_action == 'increase' and change_type == 'number' \
                    and (change_modifier == 'at least' or change_modifier == 'more than'):
                mr = 'x{} - x1 >= {}'.format(phase+1, change_extent)
            elif change_action == 'decrease' and change_type == 'percentage' \
                    and (change_modifier == 'at least' or change_modifier == 'more than'):
                mr = 'x{}/x1<=(1-{})'.format(phase+1, int(change_extent[:-1])/100.0)
            elif change_action == 'increase' and change_type == 'percentage' \
                    and (change_modifier == 'at least' or change_modifier == 'more than'):
                mr = 'x{}/x1>=(1+{})'.format(phase+1, int(change_extent[:-1])/100.0)
            elif change_action == 'decrease' and change_type == 'number' \
                    and change_modifier == 'less than':
                mr = 'x1 - x{} < {} and x1 > x{}'.format(phase+1, change_extent, phase+1)
            elif change_action == 'decrease' and change_type == 'percentage' \
                    and change_modifier == 'less than':
                mr = 'x{}/x1>(1-{}) and x1 > x{}'.format(phase+1, int(change_extent[:-1])/100.0, phase+1)
            elif change_action == 'increase' and change_type == 'number' \
                    and change_modifier == 'less than':
                mr = 'x{} - x1 < {} and x{} > x1'.format(phase+1, change_extent, phase+1)
            elif change_action == 'increase' and change_type == 'percentage' \
                    and change_modifier == 'less than':
                mr = 'x{}/x1<(1+{}) and x{} > x1'.format(phase+1, int(change_extent[:-1])/100.0, phase+1)
            elif change_action == 'stay' and change_subject == 'steering':
                mr = 'abs(x1-x{})<=1.39'.format(phase+1)
            elif change_action == 'stay' and change_subject == 'speed':
                mr = 'abs(x1-x{})<=0.1*x1'.format(phase+1)

        elif phase == 2:
            if change_action == 'decrease' and change_comparative == 'more' and change_extent is None:
                mr = 'x3 - x2 < 0'
            elif change_action == 'increase' and change_comparative == 'more' and change_extent is None:
                mr = 'x3 - x2 > 0'

        return mr

    def identify_change_action(self, root_verb):
        # print(root_verb)
        max_sim = 0
        action = None
        for t in self.config['change']:
            sim = self.get_wup_similarity(root_verb, t, 'verb')
            if sim > max_sim:
                max_sim = sim
                action = t

        return action

    def identify_transformation(self, root_verb):
        max_sim = 0
        transformation = None
        for t in self.config['transformation']:
            sim = self.get_similarity(root_verb, t)
            if sim > max_sim:
                max_sim = sim
                transformation = t

        # if transformation == 'change':
        #     transformation = 'replace'

        return transformation

    def identify_ontology(self, noun):
        if noun in self.ontology_list.keys():
            return noun
        else:
            ontology_sims = {}
            max_sim = 0
            ontology = None
            for t in self.ontology_list.keys():
                sim = self.get_wup_similarity(noun, t)
                ontology_sims[t] = sim
                if sim > max_sim:
                    max_sim = sim
                    ontology = t

            if max_sim >= 0.75:
                return ontology
            else:
                return None

    def rule_parse(self, rule_text):
        sentences = rule_text.split("\n")
        sentences = [s.split(':')[1] for s in sentences if s.strip() != '']
        sentences = [s.strip() for s in sentences if s.strip() != '']
        # objects = []
        # transformations = []
        # transformation_parameters = []
        # actions = []
        transformations = []
        mrs = []
        for i, sentence in enumerate(sentences):
            if i % 2 == 0:
                transformation = self.parse_if(sentence, phase=i // 2 + 1)
                if i // 2 + 1 == 2 and 'closer' in transformation:
                    transformations.append(tuple(list(transformations[0]) + ['closer']))
                else:
                    transformations.append(transformation)

            else:
                expected_change = self.parse_then(sentence, phase=i // 2 + 1)
                mr = self.create_mr(expected_change, phase=i // 2 + 1)
                mrs.append(mr)

        target_objects = []
        target_transformations = []
        target_parameters = []
        work_engines = []

        for transformation in transformations:
            support_engine = self.match_engine(transformation)
            if support_engine:
                work_engines.append(support_engine)
                target_transformations.append(transformation[0])
                if transformation[0] == 'add':
                    target_objects.append([transformation[1]])
                    target_parameters.append([transformation[2], transformation[-1]])
                elif transformation[0] == 'remove':
                    target_objects.append([transformation[1]])
                    target_parameters.append([])
                elif transformation[0] == 'replace':
                    if len(transformation) > 3:
                        target_objects.append([transformation[1], transformation[2]])
                    else:
                        target_objects.append([transformation[1]])
                    target_parameters.append([transformation[2]])

            else:
                work_engines.append(None)

        return target_transformations, target_objects, target_parameters, mrs, work_engines