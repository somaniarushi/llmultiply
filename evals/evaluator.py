import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Any
from multiprocessing.pool import ThreadPool
from enum import Enum

from models.base import BaseModel

NUMBERS_CSV_PATH = "data/numbers.csv"

class EvaluationType(Enum):
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"

    def from_str(s: str) -> "EvaluationType":
        if s == "addition":
            return EvaluationType.ADDITION
        elif s == "subtraction":
            return EvaluationType.SUBTRACTION
        elif s == "multiplication":
            return EvaluationType.MULTIPLICATION
        elif s == "division":
            return EvaluationType.DIVISION
        else:
            raise ValueError(f"Invalid EvaluationType: {s}")

def map_with_progress(f: callable, xs: List[Any], num_threads: int = 50):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    if num_threads == 1:
        return list(tqdm(map(f, xs), total=len(xs)))

    with ThreadPool(min(num_threads, len(xs))) as pool:
        return list(tqdm(pool.imap(f, xs), total=len(xs)))

class Evaluator:
    def __init__(self, data: str = NUMBERS_CSV_PATH):
        self.data = data

    # This can be overriden for another prompt structure like chain-of-thought, few-shot prompting...
    def get_prompt(self, number1: int, number2: int, eval_type: EvaluationType) -> str:
        """
        Generate a prompt from two numbers and the evaluation type.
        """
        prompt = ""
        if eval_type == EvaluationType.ADDITION:
            prompt = f"What is {number1} plus {number2}?"
        elif eval_type == EvaluationType.SUBTRACTION:
            prompt = f"What is {number1} minus {number2}?"
        elif eval_type == EvaluationType.MULTIPLICATION:
            prompt = f"What is {number1} times {number2}?"
        elif eval_type == EvaluationType.DIVISION:
            prompt = f"What is {number1} divided by {number2}?"

        prompt += "Think step by step and return your answer as Answer: <answer>."
        return prompt

    def get_target(self, number1: int, number2: int, eval_type: EvaluationType) -> str:
        """
        Get the target from two numbers and the evaluation type.
        """
        if eval_type == EvaluationType.ADDITION:
            return str(number1 + number2)
        elif eval_type == EvaluationType.SUBTRACTION:
            return str(number1 - number2)
        elif eval_type == EvaluationType.MULTIPLICATION:
            return str(number1 * number2)
        elif eval_type == EvaluationType.DIVISION:
            return str(number1 / number2)

    def from_type(self, eval_type: EvaluationType) -> str:
        return self.data # Might need to diversify per task later

    # This can be overriden for another evaluation metric
    def evaluate_generation(self, generation: str, target: str) -> float:
        """
        If the target is present in the generation, we return 1.0, otherwise we return 0.0.
        """
        return 1.0 if target in generation else 0.0

    def evaluate_instance(self, instance: pd.Series, generate_func: callable, eval_type: EvaluationType) -> tuple[str, float]:
        """
        Evaluate a single instance.
        """
        number1, number2 = instance["number1"], instance["number2"]
        prompt = self.get_prompt(number1, number2, eval_type)
        target = self.get_target(number1, number2, eval_type)

        generation = generate_func(prompt)
        return generation, self.evaluate_generation(generation, target)

    def evaluate(self, generate_func: callable, outdir: str, eval_type: List[EvaluationType], max_iter: Optional[int] = None) -> Dict[str, float]:
        """
        Generate a text from a prompt using the model and evaluate it against the target.
        """
        # get the corresponding csv path
        csv_path = [(t, self.from_type(t)) for t in eval_type]
        # read the csvs
        for eval_type, path in csv_path:
            df = pd.read_csv(path)
            iterations = min(len(df), max_iter) if max_iter else len(df)
            # decrease size of df
            df = df.sample(iterations)

            print(f"Starting model on {path} for {iterations} iterations")
            # evaluate the model
            list_of_tuples = map_with_progress(
                lambda i: self.evaluate_instance(df.iloc[i], generate_func, eval_type),
                range(iterations),
            )
            generations = [t[0] for t in list_of_tuples]
            scores = [t[1] for t in list_of_tuples]
            # Add scores to df
            df["generation"] = generations
            df["score"] = scores
            # Print score metadata
            print(f"Mean score: {sum(scores) / len(scores)} | Number of correct generations: {sum(scores)}")
            df.to_csv(f"{outdir}/{eval_type.value}.csv", index=False)