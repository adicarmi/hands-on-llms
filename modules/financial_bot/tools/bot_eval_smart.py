import logging
import json

import fire


from datasets import Dataset

from tools.bot import load_bot

logger = logging.getLogger(__name__)

# Path to the folder to track all changes to the local folder
code_folder_path = "../../"

EXPERIMENT_NAME = "exp6"



def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str, metrics: list) -> dict:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    from ragas import evaluate
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truths": [[ground_truth]],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)
    # Import ragas only after loading the environment variables inside load_bot()
    from ragas.metrics import (
        answer_correctness,
        answer_similarity,
        #context_entity_recall,
        context_recall,
        context_relevancy,
        #context_utilization,
        faithfulness
    )
    from ragas.metrics.context_precision import context_relevancy
    metrics = [
        #context_utilization,
        context_relevancy,
        context_recall,
        answer_similarity,
        #context_entity_recall,
        answer_correctness,
        faithfulness
    ]
    
    # import comet_ml
    # exp = comet_ml.start(project_name="adi-eval-logging")
    # exp.log_code(folder=code_folder_path)

    results = []
    total_scores = {
        "context_relevancy": 0.0,
        "context_recall": 0.0,
        "answer_similarity": 0.0,
        "faithfulness": 0.0,
        "answer_correctness": 0.0
    }

    with open(testset_path, "r") as f:
        data = json.load(f)

        for elem in data:
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": [],
            }

            output_context = bot.finbot_chain.chains[0].run(input_payload)
            response = bot.answer(**input_payload)

            score = evaluate_w_ragas(
                query=elem["question"],
                context=output_context.split('\n'),
                output=response,
                ground_truth=elem["response"],
                metrics=metrics
            )

            for key in total_scores:
                total_scores[key] += score.get(key, 0.0)

            out_payload ={
                "about_me": input_payload["about_me"],
                "question": input_payload["question"],
                "output_context": output_context,
                "response": response,
                **score}
            
            logger.info("Score =%s", out_payload)
            results.append(out_payload)

    if results:
        averaged_scores = {key: total / len(results) for key, total in total_scores.items()}
        # exp.log_metrics(averaged_scores)
        logger.info("Averaged Scores: %s", averaged_scores)

        with open(f'{EXPERIMENT_NAME}_eval_output.json', 'w') as f:
            json.dump({"scores":results, "averaged_scores":averaged_scores}, f)
    # exp.end()
    return results

if __name__ == "__main__":
    fire.Fire(run_local)
