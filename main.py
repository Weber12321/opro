
import logging
import re
import pandas as pd
import torch

from Prompts import meta_prompt, scorer_prompt
from langchain import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline


def create_chain_from_template(
    template, 
    input_variables, 
    temperature=0.5,
):
    logging.info(f"Creating chain from template: {template}")

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template
    )

    logging.info("Read model")
    hf = HuggingFacePipeline(pipeline=
        pipeline(
            task="text-generation",
            model="/opt/llm/models/Breeze-7B-32k-Instruct-v1_0" ,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
            max_new_tokens=8000,
            device_map="auto",
            temperature=temperature,
            do_sample=True,
            return_full_text=False,
        )
    ) 

    chain = prompt | hf | StrOutputParser()

    return chain

def build_text_and_scores(performance_df):
    return ''.join([f"text:\n{performance_df.iloc[i]['text']}\nscore:\n{performance_df.iloc[i]['score']}\n" for i in range(len(performance_df))])

def rank_instructions(performance_df,num_scores):
    performance_df = performance_df.sort_values(by='score')
    if len(performance_df) > num_scores:
        performance_df = performance_df.tail(num_scores)
    return performance_df

def sample_exemplars(training_exemplar_df,num_exemplars=3):
    exemplar_sample = training_exemplar_df.sample(num_exemplars)
    return ''.join([f"input:\nQ: {exemplar_sample.iloc[i]['question']}\nA: <INS>\noutput:\n{exemplar_sample.iloc[i]['raw_answer']}\n" for i in range(num_exemplars)])

def generate_prompts(optimizer_chain,texts_and_scores,exemplars,n_prompts=3):
    return [optimizer_chain.invoke(
        {"texts_and_scores": texts_and_scores, "exemplars": exemplars},
        ).replace("[","").replace("]","") for _ in range(n_prompts)]

def are_numbers_the_same(x,y):
    return ''.join(re.findall(r'\d+',x)) == ''.join(re.findall(r'\d+',y))

def score_prompts(scorer_chain,prompts,training_examples,performance_df):
    for prompt in prompts:
        scores = []
        for _, example in training_examples.iterrows():
            question = example['question']
            answer = example['raw_answer']
            sample_answer = scorer_chain.invoke({
                "question":question,"instruction":prompt
            })
            scores.append(are_numbers_the_same(answer,sample_answer))
        score = int(100*sum(scores)/len(scores))
        performance_df = performance_df._append({'text':prompt,'score':score},ignore_index=True)
    return performance_df

def opro(optimizer_chain,scorer_chain,performance_df,training_exemplar_df,n_scores=20,n_exemplars=3,n_prompts=8,n_training_samples=10,max_iterations=3):
    performance_df = rank_instructions(performance_df,n_scores)
    for _ in range(max_iterations):
        texts_and_scores = build_text_and_scores(performance_df)
        exemplars = sample_exemplars(training_exemplar_df,n_exemplars)
        prompts = generate_prompts(optimizer_chain,texts_and_scores,exemplars,n_prompts)
        training_examples = training_exemplar_df.sample(n_training_samples)
        performance_df = score_prompts(scorer_chain,prompts,training_examples,performance_df)
        performance_df = rank_instructions(performance_df,n_scores)
    return performance_df


if __name__ == '__main__':
    optimizer_chain = create_chain_from_template(
        meta_prompt,
        ["texts_and_scores","exemplars"],
        temperature=0.5,
    )
    scorer_chain = create_chain_from_template(
        scorer_prompt,
        ["question","instruction"],
        temperature=0.0,
    )
    performance_df = pd.read_csv("data/performance.csv",index_col=0)

    training_exemplar_df = pd.read_csv("data/training_exemplars.csv",index_col=0)

    output = opro(optimizer_chain,scorer_chain,performance_df,training_exemplar_df,n_scores=20,n_exemplars=3,n_prompts=8,n_training_samples=10,max_iterations=5)
    print(output)
    output.to_csv("data/performance2.csv")