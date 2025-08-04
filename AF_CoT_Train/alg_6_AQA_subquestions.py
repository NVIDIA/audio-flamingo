# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import json
import soundfile as sf
import time
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from utils import inference_qwen2_5_omni, inference_qwen3, load_models


def LLM_step_by_step(model, tokenizer, question):
    prompt = "Imagine you are listening to an audio."
    prompt = prompt + "\nDetermine if the following question about the audio requires step-by-step reasoning: {}".format(question)
    prompt = prompt + "\nReturn a json output with keys 'require_step_by_step' and 'sub_questions'."
    prompt = prompt + "\nIf the question is simple (e.g. about recognizing the source of the sound, whether a specific sound exists, or any question that could be answered by system one), the value for 'require_step_by_step' is 0, and the value for 'sub_questions' is the empty list."
    prompt = prompt + "\nIf the question does require step-by-step reasoning involving at lease two to three steps of analysis, break the reasoning process into sub questions. In this case, the value for 'require_step_by_step' is 1, and the value for 'sub_questions' is the list of sub questions."
    qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=True)["content:"]
    qwen3_output = json.loads(qwen3_output)
    return qwen3_output


def LLM_create_options(model, tokenizer, question, answer):
    if answer.lower() in ["yes", "no"]:
        choices = ["yes", "no"]
        return choices
    
    if len(answer.split(' ')) > 5:
        # not multiple choice anymore
        return None 
    
    prompt = "Imagine you are listening to an audio."
    prompt = prompt + "\nYou are asked the following question about the audio: {}".format(question)
    prompt = prompt + "\nThe ground truth answer is: {}".format(answer)
    prompt = prompt + "\nCreate three other options for the question such that they also make sense to the question and can be clearly distinguished from the ground truth answer."
    prompt = prompt + "\nFor instance, the question asks about the habitat of the animal, the ground truth is 'on the tree', then the other options can be 'in the lake', 'in the ocean', and 'on the ground'."
    prompt = prompt + "\nReturn a json output with key 'choices', and the value is the list of other options in choices."
    qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=True)["content:"]
    qwen3_output = json.loads(qwen3_output)['choices']
    if answer in qwen3_output:
        return qwen3_output
    else:
        return [answer] + qwen3_output


def LLM_create_summary(model, tokenizer, question, CoT_questions):
    prompt = "Imagine you are listening to an audio."
    prompt = prompt + "\nYou are asked the following question about the audio: {}".format(question)
    prompt = prompt + "\nHere is a possible reasoning chain to answer this question:"
    for sub_q in CoT_questions:
        prompt = prompt + "\n  - {}".format(sub_q)
    prompt = prompt + "\nBriefly summarize the question and how to think based on the provided reasoning chain."
    prompt = prompt + "\nReturn a json output with key 'summary' and value to be the summarization text."
    prompt = prompt + "\nFor instance, the question is 'Is there only one bird sqauwking?'. The reasoning chain contains two questions: 'Does the audio include squawking sounds or only chirping?', and 'How many birds are squawking in the audio?'."
    prompt = prompt + "\nThe summarization text is 'The question wants to determine whether there is one or more bird sqauwking. I will first check if the squawking sounds exist. If yes, I will then check the number of birds squawking.'."

    qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=True)["content:"]
    qwen3_output = json.loads(qwen3_output)
    return qwen3_output


def LLM_create_reasoning(model, tokenizer, question, CoT):
    prompt = "Imagine you are listening to an audio."
    prompt = prompt + "\nYou are asked the following question about the audio: {}".format(question)
    prompt = prompt + "\nHere is a possible reasoning chain to answer this question:"
    for sub_q, sub_a in CoT:
        prompt = prompt + "\n  - {} Answer: {}".format(sub_q, sub_a)
    prompt = prompt + "\nRefine the reasoning chain into a more logical, precise, and progressive reasoning chain. Disregard useless questions in the provided reasoning chain."
    prompt = prompt + "\nReturn a json output with key 'reasoning' and value to be the reasoning text."

    qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=True)["content:"]
    qwen3_output = json.loads(qwen3_output)
    return qwen3_output 


def format_question_with_choices(question, choices):
    if choices is None:
        return question 
    formatted_question = question + " Choose the correct option from the following options:"
    names = 'ABCDEFG'
    for i in range(len(choices)):
        formatted_question = formatted_question + "\n({}) {}".format(names[i], choices[i])
    return formatted_question


def format_question_no_choices(question):
    # remove the choices from the question
    for pattern in ['(a)', '(A)', 'A.']:
        if pattern in question:
            question = question.split(pattern)[0].strip()
    question = question.strip().strip('\n').strip()
    return question


def remove_option(answer):
    no_option_answer = answer.lower()
    for name in "abcdefgh":
        for pattern in ['({})'.format(name), '{}.'.format(name), '{})'.format(name), '{}:'.format(name)]:
            if pattern in no_option_answer.split(' ')[0]:
                no_option_answer = ' '.join(no_option_answer.split(' ')[1:])
                return no_option_answer
    return no_option_answer


def get_options_from_question(question):
    question = question.lower().strip()
    options = []
    names = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '\n']
    for i in range(len(names)-1):
        this_name = names[i]
        next_name = names[i + 1]
        if this_name in question:
            option = question.split(this_name)[-1].split(next_name)[0].replace('\n', '').replace('.', '').strip()
            options.append(option)
        else:
            break
    if options == []:
        options = None
    return options


# Main function
def generate_CoT(model, tokenizer, model_omni, processor_omni, audio_path, question, answer, sub_questions, verbose=False):

    S = ["Generate caption for this audio input."] + sub_questions
    C = []
    for s in S:
        omni_output = inference_qwen2_5_omni(model_omni, processor_omni, audio_path, s)
        C.append((s, omni_output))

    def LLM_predict(question, current_C):
        prompt = "Imagine you are listening to an audio."
        prompt = prompt + "\nThe caption of the audio is: {}".format(current_C[0][-1])

        if len(current_C) > 1:
            prompt = prompt + "\nThere is also a reasoning chain containing a few questions and answers, which provide more information about the audio."
            for row in current_C[1:]:
                prompt = prompt + "\nQuestion: {}".format(row[0])
                prompt = prompt + "\nAnswer: {}".format(row[1])
        
        prompt = prompt + "\nTry your best to answer the following question about the audio: {}".format(question)
        prompt = prompt + "\nReturn a json output with key 'answer'. The 'answer' is 'unsure' if it is hard to answer based on available information, and your best answer (in plain text, not option) if you could make a confident prediction."

        qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=True)["content:"]
        qwen3_output = json.loads(qwen3_output)

        return qwen3_output


    def LLM_validate(question, answer, C):
        prompt = "Imagine you are listening to an audio."
        prompt = prompt + "\nThe caption of the audio is: {}".format(C[0][-1])

        assert len(C) > 1
        prompt = prompt + "\nThere is also a reasoning chain containing a few questions and answers, which provide more information about the audio."
        for row in C[1:]:
            prompt = prompt + "\nQuestion: {}".format(row[0])
            prompt = prompt + "\nAnswer: {}".format(row[1])
        
        prompt = prompt + "\nNow, you are asked the following question about the audio: {} The ground truth answer is: {}.".format(question, answer)
        if prompt.endswith('..'):
            prompt = prompt[:-1]
        prompt = prompt + "\nDetermine if the ground truth answer can be inferred from the provided caption and reasoning chain, where 0 means unable to infer and 1 means able to infer."
        prompt = prompt + "\nReturn a json output with key 'validation' and value to be 1 or 0."

        qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=True)["content:"]
        qwen3_output = json.loads(qwen3_output)

        return qwen3_output


    try:
        LLM_prediction = LLM_predict(question, C)["answer"]
    except Exception as e:
        print("Error in LLM_predict:", e)
        LLM_prediction = 'unsure'

    if len(C) == 1:
        LLM_validation = 0
    else:
        try:
            LLM_validation = LLM_validate(question, answer, C)["validation"]
        except Exception as e:
            print("Error in LLM_validate:", e)
            LLM_validation = 0 

    if verbose:
        for row in C:
            print(row)
        print('LLM_prediction:', LLM_prediction)
        print('LLM_validation:', LLM_validation)
        print('ground truth:', answer)

    if LLM_prediction.lower().endswith(remove_option(answer)) or LLM_validation in [1, "1", True, "true", "True"]:
        is_valid_CoT_training_data = True 
    else:
        is_valid_CoT_training_data = False

    return C, LLM_prediction, LLM_validation, is_valid_CoT_training_data


# Main function
def LLM_rephrase_LLaVACoT(model, tokenizer, formatted_question, answer, CoT, verbose=False):

    if len(CoT) <= 2:
        # one caption and one question, not enough as a chain
        return None 

    full_output = ""

    # SUMMARY
    CoT_questions = [x[0] for x in CoT[1:]]
    try:
        LLM_summary = LLM_create_summary(model, tokenizer, formatted_question, CoT_questions)['summary']
    except Exception as e:
        print("Error in LLM_create_summary:", e)
        return None
    full_output = full_output + "<SUMMARY> {} </SUMMARY>".format(LLM_summary)
    if verbose:
        print(full_output)

    # CAPTION
    if (CoT[0][1] == "None") or (CoT[0][1] == "") or (len(CoT[0][1].split(' ')) < 3):
        return None
    full_output = full_output + "\n<CAPTION> {} </CAPTION>".format(CoT[0][1])

    # REASONING
    try:
        LLM_reasoning = LLM_create_reasoning(model, tokenizer, formatted_question, CoT)['reasoning']
    except Exception as e:
        print("Error in LLM_create_reasoning:", e)
        return None
    full_output = full_output + "\n<REASONING> {} </REASONING>".format(LLM_reasoning)
    if verbose:
        print(full_output.split('\n')[-1])

    # CONCLUSION
    full_output = full_output + "\n<CONCLUSION> {} </CONCLUSION>".format(answer)

    return full_output


def main(audio_path, question, answer, choices):
    output_path = "./alg_6_outputs"
    os.makedirs(output_path, exist_ok=True)

    model, tokenizer, model_omni, processor_omni = load_models()
    formatted_question = format_question_with_choices(question, choices)

    # determine if the question requires step by step
    is_valid_CoT_training_data = True
    try:
        qwen3_output = LLM_step_by_step(model, tokenizer, formatted_question)
        if 'require_step_by_step' not in qwen3_output:
            is_valid_CoT_training_data = False
        elif qwen3_output['require_step_by_step'] in [0, False, '0', 'False', 'false']:
            is_valid_CoT_training_data = False
        elif 'sub_questions' not in qwen3_output:
            is_valid_CoT_training_data = False
        elif type(qwen3_output['sub_questions']) is not list or len(qwen3_output['sub_questions']) == 0:
            is_valid_CoT_training_data = False
    except Exception as e:
        print('Error in LLM_step_by_step:', e)
        is_valid_CoT_training_data = False
    
    if not is_valid_CoT_training_data:
        print("The question '{}' does not require step-by-step reasoning to answer.".format(question))
        return None
    
    # generate reasoning chains
    sub_questions = qwen3_output['sub_questions']
    C, LLM_prediction, LLM_validation, is_valid_CoT_training_data = generate_CoT(
        model, tokenizer, model_omni, processor_omni, 
        audio_path, formatted_question, answer, sub_questions
    )

    if not is_valid_CoT_training_data:
        print("The reasoning chain is not valid.")
        return None
    
    # turn reasoning chain (C) into LLaVACoT template
    LLaVACoT_output = LLM_rephrase_LLaVACoT(model, tokenizer, formatted_question, answer, C)

    output_dic = {
        "audio_path": audio_path,
        "question": question,
        "answer": answer,
        "choices": choices,
        "formatted_question": formatted_question,
        "sub_questions": sub_questions,
        "LLaVACoT_output": LLaVACoT_output,
    }
    return output_dic


if __name__ == '__main__':
    audio_path = "path/to/audio.wav"
    question = "Your question about the audio"
    answer = "ground truth answer"
    choices = ['choice A', 'choice B', 'choice C', 'choice D']  # or None if it is not a multiple choice question

    output_dic = main(audio_path, question, answer, choices)
    print(output_dic)