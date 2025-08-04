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
def generate_CoT(model, tokenizer, model_omni, processor_omni, audio_path, question, answer, choices, verbose=False):

    S = ["Generate caption for this audio input."]
    for c in choices:
        S.append("Describe the acoustic properties of '{}' in one or two sentences. Just focus on the information in terms of sound.".format(c))
        S.append("How does this sound fit with the label '{}'?".format(c))
    C = []
    for s in S:
        if s.startswith("Describe"):
            omni_output = inference_qwen2_5_omni(model_omni, processor_omni, None, s)
        else:
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

        qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=False)["content:"]
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

        qwen3_output = inference_qwen3(model, tokenizer, prompt, record_time=False)["content:"]
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


def Rule_create_summary(choices):
    summary = "The question is to classify the sound to be one of {}. For each choice, I will describe its acoustic properties, and then decide whether the sound fits that label.".format(
        ", ".join(choices)
    )
    return summary


def Rule_create_reasoning(CoT):
    reasoning = ""
    n_CoT = len(CoT)
    assert n_CoT % 2 == 1
    for i in range(1, n_CoT, 2):
        reasoning += CoT[i][1] + " " + CoT[i+1][1] + "\n"
    return reasoning


# Main function
def Rule_rephrase_LLaVACoT(choices, answer, CoT):

    if len(CoT) <= 2:
        # one caption and one question, not enough as a chain
        return None 

    full_output = ""

    # SUMMARY
    Rule_summary = Rule_create_summary(choices)
    full_output = full_output + "<SUMMARY> {} </SUMMARY>".format(Rule_summary)

    # CAPTION
    if (CoT[0][1] == "None") or (CoT[0][1] == "") or (len(CoT[0][1].split(' ')) < 3):
        return None
    full_output = full_output + "\n<CAPTION> {} </CAPTION>".format(CoT[0][1])

    # REASONING
    Rule_reasoning = Rule_create_reasoning(CoT)
    full_output = full_output + "\n<REASONING> {} </REASONING>".format(Rule_reasoning)

    # CONCLUSION
    full_output = full_output + "\n<CONCLUSION> {} </CONCLUSION>".format(answer)

    return full_output


def main(audio_path, question, answer, choices):
    output_path = "./alg_8_outputs"
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
    except Exception as e:
        print('Error in LLM_step_by_step:', e)
        is_valid_CoT_training_data = False
    
    if not is_valid_CoT_training_data:
        print("The question '{}' does not require step-by-step reasoning to answer.".format(question))
        return None
    
    # generate reasoning chains
    C, LLM_prediction, LLM_validation, is_valid_CoT_training_data = generate_CoT(
        model, tokenizer, model_omni, processor_omni, 
        audio_path, formatted_question, answer, choices
    )

    if not is_valid_CoT_training_data:
        print("The reasoning chain is not valid.")
        return None
    
    # turn reasoning chain (C) into LLaVACoT template
    LLaVACoT_output = Rule_rephrase_LLaVACoT(choices, answer, C)

    output_dic = {
        "audio_path": audio_path,
        "question": question,
        "answer": answer,
        "choices": choices,
        "formatted_question": formatted_question,
        "LLaVACoT_output": LLaVACoT_output,
    }
    return output_dic


if __name__ == '__main__':
    audio_path = "path/to/audio.wav"
    question = "Classify the sound."  # default question
    answer = "ground truth answer"
    choices = ['choice A', 'choice B', 'choice C', 'choice D']  # or None if it is not a multiple choice question

    output_dic = main(audio_path, question, answer, choices)
    print(output_dic)