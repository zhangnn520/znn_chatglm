#!/usr/bin/env python
# coding=utf-8
# Implement several parameter-efficient fine-tuning method for ChatGLM.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
import os
import sys
import json

sys.path.append('./src/utils')

from utils import (
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    plot_loss,
    DataCollatorForChatGLM,
    ComputeMetrics,
    TrainerForChatGLM
)


def main():
    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    # 默认训练和验证集必须一起出现，后续训练部分传入参数参看原始finetune.sh，里面已经配置好了训练、验证、保存模型的参数。
    # dataset_one 第一个值默认为训练集或测试集，dataset_two 非空默认为验证集
    dataset_one, dataset_two = prepare_data(model_args, data_args, training_args)
    model, tokenizer = load_pretrained(model_args, finetuning_args, is_trainable=training_args.do_train)
    if training_args.do_train and training_args.do_eval:
        train_dataset = preprocess_data(dataset_one, tokenizer, data_args, training_args)
        eval_dataset = preprocess_data(dataset_two, tokenizer, data_args, training_args)
        test_dataset = None
    else:
        train_dataset = None
        eval_dataset = None
        test_dataset = preprocess_data(dataset_one, tokenizer, data_args, training_args)

    data_collator = DataCollatorForChatGLM(tokenizer, model, data_args.ignore_pad_token_for_loss, training_args.do_eval)

    # Override the decoding parameters of Trainer
    training_args.generation_max_length = training_args.generation_max_length if \
        training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.num_beams if \
        data_args.num_beams is not None else training_args.generation_num_beams
    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        raise AssertionError("请确认您需要做训练、验证还是预测，必须至少存在一个任务才能正常进行代码")

    # Initialize our Trainer
    trainer = TrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None
    )

    if training_args.do_train:
        # 这里trainer.train中已经可以包含训练和验证功能，所有没有必要再写验证部分的代码
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if finetuning_args.plot_loss:
            plot_loss(training_args)

    if training_args.do_predict:
        # Predict
        model.half().cuda()
        predict_results = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test", do_sample=True,
                                          top_p=0.7, max_length=768, temperature=0.95)
        labels = tokenizer.batch_decode(
            predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        labels = [label.strip() for label in labels]
        output_prediction_file = os.path.join(model_args.checkpoint_dir[0], "generated_predictions.json")
        with open(output_prediction_file, "w+", encoding="utf-8") as writer:
            for p, l in zip(predictions, labels):
                res = json.dumps({"答案": l, "预测": p}, ensure_ascii=False, indent=4)
                writer.write(f"{res}\n")
        trainer.log_metrics("test", predict_results.metrics)
        trainer.save_metrics("test", predict_results.metrics)


if __name__ == "__main__":
    main()
