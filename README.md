# bert-based-text-generation
Text generation by iterative word replacement.

## How does it work?
BERT based Text Generation applies one of the two pretraining steps of BERT, masked word prediction, for text generation.
Masked word prediction in BERT pretraining looks like:
```
Masked input: the man went to the [MASK] .
Prediction: [MASK] = store
(Modified from README.md in bert)
```

Actually, the model outputs confidence for every word in the dictionary. Thus, by choosing the “incorrect” word with the highest confidence, one can replace a word in a sentence hopefully without any grammatical or semantic incorrectness.


Here is a single step of word replacement at a random position.
```
Input: the man went to the store .
Masked input: the man went to the [MASK] .
Output: the man went to the office .
```


And if you apply the step iteratively, starting from a “seed” sentence, until all words are replaced:
```
Seed: the man went to the store .
Masked input 1: the man went to the [MASK] .
Output 1: the man went to the office .
Masked input 2: the man [MASK] to the office .
Output 2: the man came to the office .
…
Output n: a cat is on the desk !
```


Now you’ve got a new sentence!

## Related studies
Generative models with iterative update has been proposed for a long time: eg, denoising auto encoders (https://arxiv.org/abs/1305.6663) and generative stochastic networks (https://arxiv.org/abs/1306.1091). I have not fully surveyed literatures but I suppose there should be many such studies in text generation, too.

## Dependencies
Same as bert-japanese.

## License
MIT.


Submodules bert-japanese and bert are released under the Apache 2.0 license.
