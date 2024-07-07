# ArithEval
An evaluation of large language models on basic arithmetic abilities for up to 20 digit operations. Currently supports addition, subtraction, multiplication and division.

## Setup and Run
To set up the environment, run 
```
pip install -r requirements.txt
```
to add dependencies to the environment. The Llama model loader uses [together API](https://www.together.ai/) to serve the Llama3 8B Chat model. A user can sign up for an account and run the model by setting the `TOGETHER_BEARER_TOKEN` in the environment. Alternatively, to run locally, simply create another model generator matching the abstraction of the base model.
```
class BaseModel(ABC):
    @abstractmethod
    def generate(
        self, 
        prompt: str,
        max_tokens_to_generate: int,
        temperature: float,
        top_k: int, 
    ) -> str:
        raise NotImplementedError(f"generate method not implemented for {self.__class__.__name__}")

```
An example on how to run the evaluation is given in `scripts/evaluate_llam8b.py`
```
from evals.evaluator import Evaluator, EvaluationType
from models.llama import LlamaSampler

sampler = LlamaSampler()
generate_func = lambda prompt: sampler.generate(prompt, max_tokens_to_generate=1000, temperature=0.5, top_k=40)
evaluator = Evaluator(data="data/numbers.csv")

# Evaluate the model on 10 steps
evaluator.evaluate(
    generate_func=generate_func,
    outdir="outputs/llama8b",
    eval_type=[EvaluationType.SUBTRACTION],
)
```
This repository is opinionated about not supposed CLI calls to run evaluations but contributors are welcome to add them if desired!

## Llama3 8B's Performance
| Operation     | Score |
|---------------|-------|
| Addition      | 36.8  |
| Subtraction   | 32.2  |
| Multiplication| 11.4  |

Here is the addition breakdown of the model performance by number length:
<image src="https://github.com/somaniarushi/llmultiply/assets/54224195/fe1d68f4-126b-45fa-a08f-d1a2e1de4f54" width=500>

Here is the subtraction breakdown of the model performance by number length: 
<image src="https://github.com/somaniarushi/llmultiply/assets/54224195/f45cf12f-37de-4d0a-ba5c-c7cf910f4846" width=500>


Here is the multiplication breakdown of the model performance by number length:
<image src="https://github.com/somaniarushi/llmultiply/assets/54224195/e1f27481-7217-4135-bcb5-02865879d930" width=500>

Read more about the experiments [here](https://dailyink.notion.site/Measuring-the-Arithmetic-Capabilities-of-Frontier-Models-fe700448b7c04c7faf61974221390ebd?pvs=74).

## Contributing
Contributions are welcome. Email hi@amks.me for collaborations or questions.

