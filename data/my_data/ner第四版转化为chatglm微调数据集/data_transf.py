# #!/usr/bin/python
# # -*- coding:utf-8 -*-
import os
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
        writer.write(json.dumps(content_line, ensure_ascii=False,indent=4))


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
        self.fine_tune_train_path = os.path.join(base_path, '../enoch_fine_tune_train.json')
        self.fine_tune_dev_path = os.path.join(base_path, '../enoch_fine_tune_dev.json')
        self.fine_tune_test_path = os.path.join(base_path, '../enoch_fine_tune_test.json')

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
        print('输入的最大长度为：', max(self.prompt_instruction_len_list))
        print('输出的最大长度为：', max(self.result_len_list))

    # @staticmethod
    def create_fine_tune_data(self, data_list, path):
        prompt_data_list = list()
        for num, datas in enumerate(data_list):
            content = datas['raw_text']
            temp_dict = dict()
            question = content
            answer = datas['prompt_answer']
            if len(question) <= max_text_length and len(answer) <= max_text_length:  # 设置最大长度
                self.prompt_instruction_len_list.append(len(question))
                temp_dict[
                    'instruction'] = "你现在是一个命名实体识别模型，请你帮我抽取出命名实体识别类别为'安装动作','操作部件位置1','方位','目标部件位置1','操作程序选项','一般动作','操作部件位置2','目标部件2','物理量','目标部件位置2','量词','工作区域','拆卸动作','操作程序','目标部件1','操作部件2','操作部件1','一般工具'的二元组，二元组内部用'_'连接，二元组之间用'&'分割。"
                temp_dict['input'] = question

                temp_dict['output'] = '\n'.join(answer)
                prompt_data_list.append(temp_dict)
                if len('\n'.join(answer))==227 or len('\n'.join(answer))==231:
                    print(question)
                if len(question)==227 or len(question)==231:
                    print(question)
                self.result_len_list.append(len('\n'.join(answer)))
        prompt_data_list = delete_duplicate_elements(prompt_data_list)
        print('句子个数为：', len(prompt_data_list))
        write_json(path, prompt_data_list)


# {
#   'instruction': '读下面的段落，找出一个比喻句子。',
#   'input': ''我的烦恼长出了翅膀，飞走进了天空'',
#   'output': '比喻：我的烦恼长出了翅膀，飞走进了天空。'
# },

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
                    entity_spans.append({'起始': start, '终止': end})
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
                    entity_spans.append({'起始': start, '终止': end})
                    start_flag = False
        if start_flag:  # 句子以实体I-结束，未被添加
            entities.append(''.join(words[start:end + 1]))
            entity_tags.append(targets[start][2:].lower())
            entity_spans.append({'起始': start, '终止': end + 1})
            start_flag = False

        output['entities'] = [(i, entities[i_index]) for i_index, i in enumerate(entity_tags)]
        output['raw_text'] = ''.join(words)
        output['entity_spans'] = [(i, entity_spans[i_index]) for i_index, i in enumerate(entity_tags)]
        # output['prompt_answer'] = [[{'类别': i}, {'名称': entities[i_index]},
        #                             {'位置': entity_spans[i_index]}
        #                             ] for i_index, i in enumerate(entity_tags)]
        output['prompt_answer'] = [f'{i}_{entities[i_index]}' for i_index, i in enumerate(entity_tags)]
        ner_label += [i for i in entity_tags]
        outputs_list.append(output)
    return outputs_list


if __name__ == '__main__':
    processor = DataProcessor()
    processor.train_eval_test_data_processor()
    print(list(set(ner_label)))
