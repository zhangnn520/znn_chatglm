# #!/usr/bin/python
# # -*- coding:utf-8 -*-
import os
import copy
import logging
import json

base_path = os.path.dirname(__file__)
logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s-%(funcName)s:%(lineno)d', level=logging.INFO)
max_text_length = 200

ner_label = []


def read_text(path):
    with open(path, 'r', encoding='utf-8') as reader:
        return [i.replace('\n', '') for i in reader.readlines()]


def write_json(write_path, content_line):
    with open(write_path, 'w+', encoding='utf-8') as writer:
        writer.write(json.dumps(content_line, ensure_ascii=False, indent=4))


def delete_duplicate_elements(list_data):
    from functools import reduce
    return reduce(lambda x, y: x if y in x else x + [y], [[], ] + list_data)


class DataProcessor:
    def __init__(self):
        self.prompt_instruction_len_list = list()
        self.result_len_list = list()
        self.train_path = os.path.join(base_path, 'ann_data', 'train.txt')
        self.dev_path = os.path.join(base_path, 'ann_data', 'dev.txt')
        self.test_path = os.path.join(base_path, 'ann_data', 'test.txt')
        self.fine_tune_dev_path = os.path.join(base_path, '../dev_examples.json')
        self.fine_tune_train_path = os.path.join(base_path, '../train_examples.json')
        self.fine_tune_test_path = os.path.join(base_path, '../test_examples.json')

    @staticmethod
    def get_ner_content(filepath):
        content_list = read_text(filepath)
        return content_list

    def train_eval_test_data_processor(self):
        ner_train_data = self.get_ner_content(self.train_path)
        ner_eval_data = self.get_ner_content(self.dev_path)
        ner_test_data = self.get_ner_content(self.test_path)
        train_data = get_words_bio_label(ner_train_data)
        eval_data = get_words_bio_label(ner_eval_data)
        test_data = get_words_bio_label(ner_test_data)
        self.create_fine_tune_data(train_data, self.fine_tune_train_path)
        self.create_fine_tune_data(eval_data, self.fine_tune_dev_path)
        self.create_fine_tune_data(test_data, self.fine_tune_test_path)
        # print('输入的最大长度为：', max(self.prompt_instruction_len_list))
        # print('输出的最大长度为：', max(self.result_len_list))

    # @staticmethod
    def create_fine_tune_data(self, data_list, path):
        instruction_data_list = list()
        for num, datas in enumerate(data_list):
            content = datas['raw_text']
            temp_dict1 = dict()
            question = content
            answers = datas['prompt_answer']
            if len(question) <= max_text_length and len(answers) <= max_text_length:  # 设置最大长度
                ner_class = [i[0][0] for i in answers]
                instruction = "你现在是一个命名实体识别模型，请你帮我判断一下文本存在那些类别，" \
                              "这些类别包括'安装动作','操作部件位置1','方位','目标部件位置1','操作程序选项','一般动作','操作部件位置2','目标部件2','物理量','目标部件位置2','量词','工作区域'," \
                              "'拆卸动作','操作程序','目标部件1','操作部件2','操作部件1','一般工具',类别之间用'$'分割。"
                temp_dict1['instruction'] = instruction
                temp_dict1['input'] = question
                temp_dict1['output'] = "$".join(ner_class)
                temp_dict1['history'] = [["",""], ["",""]]
                instruction_data_list.append(temp_dict1)
                ner_word_dict = {i: [] for i in ner_class}
                start_and_end_dict = {j: [] for j in ner_class}
                for num, ner_label in enumerate(ner_class):
                    try:
                        ner_word = [i[0][1] for i in answers][num]
                        start_and_end = [i[0][2] for i in answers][num]
                        if ner_label in ner_word_dict.keys():
                            new_word_list = copy.deepcopy(ner_word_dict[ner_label])
                            ner_word_dict[ner_label] = new_word_list + [ner_word]
                        if ner_label in start_and_end_dict.keys():
                            new_start_and_end_list = copy.deepcopy(start_and_end_dict[ner_label])
                            start_and_end_dict[ner_label] = new_start_and_end_list + [start_and_end]
                    except Exception as e:
                        print(ner_label)
                for num, ner_word_key in enumerate(ner_word_dict):
                    temp_dict2 = dict()
                    temp_dict3 = dict()
                    new_start_and_end_list = list()
                    ner_word_instruction = f"你现在是一个命名实体识别模型，请你帮我判断一下句子中，名词类别为'{ner_word_key}'有哪些名词？,用'$'分割"
                    temp_dict2['instruction'] = ner_word_instruction
                    temp_dict2['input'] = question
                    temp_dict2['output'] = "$".join(ner_word_dict[ner_word_key])
                    temp_dict2['history'] = [[instruction + "文本为：" + question, "$".join(ner_class)],["",""]]
                    instruction_data_list.append(temp_dict2)
                    ner_word_result = "$".join(ner_word_dict[ner_word_key])
                    words_string = "$".join(ner_word_dict[ner_word_key])
                    temp_dict3['instruction'] = f"你现在是一个命名实体识别模型，请你帮我判断一下句子中，名词为'{words_string}',起始位置和终止位置分别是什么？,用'$'分割"
                    temp_dict3['input'] = question
                    temp_dict3['history'] = [[instruction + "文本为：" + question, "$".join(ner_class)],
                                             [ner_word_instruction + "文本为：" + question, ner_word_result]]

                    for num1, start_word_index in enumerate(
                            ["起始位置:" + str(i[0]) for i in start_and_end_dict[ner_word_key]]):
                        temp_string = ["终止位置:" + str(i[1]) for i in start_and_end_dict[ner_word_key]]
                        new_start_and_end_value = start_word_index + ","+temp_string[0]
                        new_start_and_end_list.append(new_start_and_end_value)
                    temp_dict3['output'] = "$".join(new_start_and_end_list)
                    # instruction_data_list.append(temp_dict3)
        write_json(path, instruction_data_list)


def get_words_bio_label(label_list, ner_label=ner_label):
    '''返回服务所需的数据格式'''
    outputs_list = []
    raw_words, raw_targets = [], []
    raw_word, raw_target = [], []
    for line in label_list:
        if line != '':
            raw_word.append(line.split('\t')[0])
            raw_target.append(line.split('\t')[1])
        else:
            raw_words.append(raw_word)
            raw_targets.append(raw_target)
            raw_word, raw_target = [], []

    for words, targets in zip(raw_words, raw_targets):
        output, entities, entity_tags, entity_spans = {}, [], [], []
        start, end, start_flag = 0, 0, False
        for idx, tag in enumerate(targets):
            if tag.startswith('B-'):  # 一个实体开头 另一个实体（I-）结束
                end = idx
                if start_flag:  # 另一个实体以I-结束，紧接着当前实体B-出现
                    entities.append(''.join(words[start:end]))
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append((start, end))
                    start_flag = False
                start = idx
                start_flag = True
            elif tag.startswith('I-'):  # 实体中间，不是开头也不是结束，end+1即可
                end = idx
            elif tag.startswith('O'):  # 无实体，可能是上一个实体的结束
                end = idx
                if start_flag:  # 上一个实体结束
                    entities.append(''.join(words[start:end]))
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append((start, end))
                    start_flag = False
        if start_flag:  # 句子以实体I-结束，未被添加
            entities.append(''.join(words[start:end + 1]))
            entity_tags.append(targets[start][2:].lower())
            entity_spans.append((start, end + 1))
            start_flag = False

        output['entities'] = [(i, entities[i_index]) for i_index, i in enumerate(entity_tags)]
        output['raw_text'] = ''.join(words)
        output['entity_spans'] = [(i, entity_spans[i_index]) for i_index, i in enumerate(entity_tags)]
        output['prompt_answer'] = [[(i, entities[i_index], entity_spans[i_index])] for i_index, i in
                                   enumerate(entity_tags)]
        # output['prompt_answer'] = [f'{i}_{entities[i_index]}' for i_index, i in enumerate(entity_tags)]
        ner_label += [i for i in entity_tags]
        outputs_list.append(output)
    return outputs_list


if __name__ == '__main__':
    processor = DataProcessor()
    processor.train_eval_test_data_processor()
