# Detecting Hate Speech with GPT-3

Sophisticated language models such as OpenAI's GPT-3 can generate hateful text that targets marginalized groups. Given this capacity, we are interested in whether large language models can be used to identify hate speech and classify text as sexist or racist? We use GPT-3 to identify sexist and racist text passages with zero-, one-, and few-shot learning. We find that with zero- and one-shot learning, GPT-3 is able to identify sexist or racist text with an accuracy between 48 per cent and 69 per cent. With few-shot learning and an instruction included in the prompt, the model's accuracy can be as high as 78 per cent. We conclude that large language models have a role to play in hate speech detection, and that with further development language models could be used to counter hate speech and even self-police.

## Data Collection

We interact with GPT-3 using the OpenAI API via the `openai` Python package. The package requires an API key and organization ID for authentication. These can be retrieved by [creating an OpenAI account](https://openai.com/api/) and inputted where prompted in the data collection scripts (under `inputs`). Within the script files, the API is accessed within the body of the function written for each type of learning. Each function takes a comment, an example or examples where applicable, and additional parameters, passes the comment and example(s) to the API, formats the API output, and returns the model's classification alongside information about the inputs. The functions are applied to different datasets using for loops, which can be executed as-is.

Use of the OpenAI API is [not free](https://openai.com/api/pricing/), however each new account is granted 18 USD of credit which should be sufficient for use of our code.

## Thanks 
We gratefully acknowledge the support of Gillian Hadfield and the Schwartz Reisman Institute for Technology and Society. We thank Amy Farrow, Haoluan Chen, Mauricio Vargas Sepúlveda, and Tom Davidson for helpful suggestions. Comments on this paper are welcome at: rohan.alexander@utoronto.ca.
