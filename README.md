# EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis

[The EmoLLMs Paper](https://arxiv.org/abs/2401.08508)

## News

📢 *Jan. 31, 2024* We release the training code of EmoLLMs models and some data examples! More datasets will be released soon.
📢 *Jan. 23, 2024* We release the series of EmoLLMs models!

## Introduction

This project presents our efforts towards comprehensive affective analysis with large language models (LLMs).
The model can be used for affective classification tasks (e.g. sentimental polarity
or categorical emotions), and regression tasks (e.g. sentiment strength or emotion intensity).

## Ethical Consideration

Recent studies have indicated LLMs may introduce some potential
bias, such as gender gaps. Meanwhile, some incorrect prediction results, and over-generalization
also illustrate the potential risks of current LLMs. Therefore, there
are still many challenges in applying the model to real-scenario
affective analysis systems.

## Models in EmoLLMs

There are a series of EmoLLMs, including Emollama-7b, Emollama-chat-7b, Emollama-chat-13b,  Emoopt-13b, Emobloom-7b, Emot5-large, Emobart-large.

- [Emollama-7b](https://huggingface.co/lzw1008/Emollama-7b): This model is finetuned based on the LLaMA2-7B. 
- [Emollama-chat-7b](https://huggingface.co/lzw1008/Emollama-chat-7b): This model is finetuned based on the LLaMA2-chat-7B.  
- [Emollama-chat-13b](https://huggingface.co/lzw1008/Emollama-chat-13b): This model is finetuned based on the LLaMA2-chat-13B. 
- [Emoopt-13b](https://huggingface.co/lzw1008/Emoopt-13b): This model is finetuned based on the OPT-13B. 
- [Emobloom-7b](https://huggingface.co/lzw1008/Emobloom-7b): This model is finetuned based on the Bloomz-7b1-mt. 
- [Emot5-large](https://huggingface.co/lzw1008/Emot5-large): This model is finetuned based on the T5-large.
- [Emobart-large](https://huggingface.co/lzw1008/Emobart-large): This model is finetuned based on the bart-large. 

All models are trained on the full AAID instruction tuning data.



## Usage

You can use the series of EmoLLMs models in your Python project with the Hugging Face Transformers library. Here is a simple example of how to load the model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
```

In this examples, AutoTokenizer is used to load the tokenizer, and AutoModelForCausalLM is used to load the model. The `device_map='auto'` argument is used to automatically
use the GPU if it's available. `MODEL_PATH` denotes your model save path.

### Prompt examples

#### Emotion intensity

    Human: 
    Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.
    Text: @CScheiwiller can't stop smiling 😆😆😆
    Emotion: joy
    Intensity Score:

    Assistant:

    >>0.896

#### Sentiment strength

    Human:
    Task: Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive).
    Text: Happy Birthday shorty. Stay fine stay breezy stay wavy @daviistuart 😘
    Intensity Score:

    Assistant:

    >>0.879

#### Sentiment classification

    Human:
    Task: Categorize the text into an ordinal class that best characterizes the writer's mental state, considering various degrees of positive and negative sentiment intensity. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred
    Text: Beyoncé resentment gets me in my feelings every time. 😩
    Intensity Class:

    Assistant:

    >>-3: very negative emotional state can be inferred

#### Emotion classification

    Human:
    Task: Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).
    Text: Whatever you decide to do make sure it makes you #happy.
    This text contains emotions:

    Assistant:

    >>joy, love, optimism
The task description can be adjusted according to the specific task.

After loading the models, you can generate a response. Here is an example:

```python
prompt = '''Human: 
Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.
Text: @CScheiwiller can't stop smiling 😆😆😆
Emotion: joy
Intensity Score:

Assistant:
'''

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs["input_ids"], max_length=256)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)
```
Batch inference. The data format needs to follow data/test.json.
```python
bash src/run_inference.sh
```

### Finetune
```python
bash src/run_sft.sh
```

## License

EmoLLMs series are licensed under [MIT]. Please find more details in the [MIT](LICENSE) file.

## Citation

If you use the series of EmoLLMs in your work, please cite our paper:

```bibtex
@article{liu2024emollms,
  title={EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis},
  author={Liu, Zhiwei and Yang, Kailai and Zhang, Tianlin and Xie, Qianqian and Yu, Zeping and Ananiadou, Sophia},
  journal={arXiv preprint arXiv:2401.08508},
  year={2024}
}
```
