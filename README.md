# BERT for Japanese Twitter

<div align="justify">

This repository describes the development of BERT for Japanese Twitter, a pretrained model specialized for Japanese SNS tasks.

The models are available on [Transformers](https://github.com/huggingface/transformers) by Hugging Face.

![Corpus Contribution vs. Population by Prefecture](geo.png)

This publication introduces novel, open-source resources for sentiment analysis on Japanese Twitter. BERT for Japanese Twitter is a pre-trained model that is highly competent in the target domain and adaptable to a variety of tasks. Japanese Twitter Sentiment 1k (JTS1k) is a compact sentiment analysis dataset optimized for balance and reliability. This combination of pre-trained model and dataset was used to fine-tune a sentiment analysis model that broadly applies to Japanese social networking services (SNS): BERT for Japanese SNS Sentiment.  

The primary focus of this project is domain adaptation. Using an established Japanese BERT model as a foundation, domain adaptation was achieved by optimizing the vocabulary and continuing pre-training on a large Twitter corpus. Similar methodology was used to develop Twitter Multilingual RoBERTa (XLM-T) (Barbieri et al., 2022), which is the state-of-the-art multilingual Twitter model. By using a monolingual approach, this study developed a more efficient model that outperformed XLM-T in the target language.  

This project explored fundamental elements of corpus construction, corpus refinement, dataset annotation, preprocessing, pre-training, fine-tuning, and benchmarking. It concludes with a demonstration that the sentiment model is valid, useful, and sensitive to changes in public sentiment that correlate with real-world events.

</div>
