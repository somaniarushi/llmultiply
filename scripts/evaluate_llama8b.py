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
    # eval_type=[EvaluationType.ADDITION],
    # max_iter=10
)