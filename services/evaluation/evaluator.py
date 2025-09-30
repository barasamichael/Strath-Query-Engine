import json
import logging
from pathlib import Path

from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config.settings import ROOT_DIR
from services.retrieval.vector_db import VectorDBService
from services.generation.intent_recognizer import IntentRecognizer
from services.generation.response_generator import ResponseGenerator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


class RAGEvaluator:
    def __init__(
        self,
        vector_db_service: Optional[VectorDBService] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.vector_db_service = vector_db_service or VectorDBService()
        self.intent_recognizer = intent_recognizer or IntentRecognizer()
        self.response_generator = response_generator or ResponseGenerator()

        self.eval_dir = ROOT_DIR / "tests" / "eval_data"
        if not self.eval_dir.exists():
            self.eval_dir.mkdir(parents=True)

    def create_eval_set(self, output_path: Optional[Path] = None) -> Path:
        """Create a sample evaluation set template."""
        output_path = output_path or self.eval_dir / "eval_questions.csv"

        # Create sample evaluation questions
        eval_questions = [
            {
                "id": "q1",
                "query": "What are the academic departments at Strathmore University?",
                "expected_intent": "factual_query",
                "expected_topic": "academics",
                "expected_answer_contains": [
                    "Business School",
                    "Computing",
                    "Humanities",
                    "Mathematical Sciences",
                    "Law School",
                ],
                "notes": "Should list all main academic departments",
            },
            {
                "id": "q2",
                "query": "How do I apply for a scholarship at Strathmore?",
                "expected_intent": "procedural_query",
                "expected_topic": "fees",
                "expected_answer_contains": [
                    "Financial Aid",
                    "application",
                    "process",
                ],
                "notes": "Should explain scholarship application process",
            },
            {
                "id": "q3",
                "query": "What is the dress code at Strathmore?",
                "expected_intent": "factual_query",
                "expected_topic": "policies",
                "expected_answer_contains": ["professional", "dress", "code"],
                "notes": "Should explain the dress code requirements",
            }
            # Add more evaluation questions as needed
        ]

        # Save as CSV
        df = pd.DataFrame(eval_questions)
        df.to_csv(output_path, index=False)

        logger.info(f"Created evaluation set template at {output_path}")
        return output_path

    def run_evaluation(
        self,
        eval_file: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on the provided evaluation set."""
        eval_file = eval_file or self.eval_dir / "eval_questions.csv"
        output_path = output_path or self.eval_dir / "eval_results.json"

        if not eval_file.exists():
            logger.warning(f"Evaluation file not found: {eval_file}")
            logger.info("Creating sample evaluation set template...")
            eval_file = self.create_eval_set()

        # Load evaluation questions
        # services/evaluation/evaluator.py (continued)
        try:
            eval_df = pd.read_csv(eval_file)
            logger.info(
                f"Loaded {len(eval_df)} evaluation questions from {eval_file}"
            )
        except Exception as e:
            logger.error(f"Error loading evaluation file: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading evaluation file: {str(e)}",
            }

        # Run evaluation on each question
        results = []
        for _, row in tqdm(
            eval_df.iterrows(), total=len(eval_df), desc="Evaluating"
        ):
            try:
                query = row["query"]

                # Recognize intent
                intent_info = self.intent_recognizer.recognize_intent(query)

                # Retrieve relevant context
                retrieved_chunks = []
                if intent_info["intent_type"] != "off_topic":
                    retrieved_chunks = self.vector_db_service.search(
                        query=query, top_k=5
                    )

                # Generate response
                response_data = self.response_generator.generate_response(
                    query=query,
                    retrieved_context=retrieved_chunks,
                    intent_info=intent_info,
                )

                # Check if response contains expected content
                expected_contains = row.get("expected_answer_contains", [])
                if isinstance(expected_contains, str):
                    # Convert string representation of list to actual list
                    expected_contains = eval(expected_contains)

                contains_expected = all(
                    term.lower() in response_data["response"].lower()
                    for term in expected_contains
                )

                # Check if intent matches expected
                intent_match = (
                    row.get("expected_intent", "")
                    == intent_info["intent_type"].value
                )
                topic_match = (
                    row.get("expected_topic", "") == intent_info["topic"].value
                )

                # Calculate score (simple version)
                score = 0
                if intent_match:
                    score += 0.25
                if topic_match:
                    score += 0.25
                if contains_expected:
                    score += 0.5

                # Store result
                result = {
                    "id": row.get("id", ""),
                    "query": query,
                    "response": response_data["response"],
                    "expected_intent": row.get("expected_intent", ""),
                    "actual_intent": intent_info["intent_type"].value,
                    "intent_match": intent_match,
                    "expected_topic": row.get("expected_topic", ""),
                    "actual_topic": intent_info["topic"].value,
                    "topic_match": topic_match,
                    "expected_answer_contains": expected_contains,
                    "contains_expected": contains_expected,
                    "score": score,
                    "token_usage": response_data.get("token_usage", {}),
                }

                results.append(result)

            except Exception as e:
                logger.error(
                    f"Error evaluating query '{row.get('query', '')}': {str(e)}"
                )
                results.append(
                    {
                        "id": row.get("id", ""),
                        "query": row.get("query", ""),
                        "error": str(e),
                        "score": 0,
                    }
                )

        # Calculate overall metrics
        total_score = sum(result.get("score", 0) for result in results)
        avg_score = total_score / len(results) if results else 0
        intent_accuracy = (
            sum(1 for result in results if result.get("intent_match", False))
            / len(results)
            if results
            else 0
        )
        topic_accuracy = (
            sum(1 for result in results if result.get("topic_match", False))
            / len(results)
            if results
            else 0
        )
        content_accuracy = (
            sum(
                1
                for result in results
                if result.get("contains_expected", False)
            )
            / len(results)
            if results
            else 0
        )

        # Aggregate token usage
        total_tokens = sum(
            result.get("token_usage", {}).get("total_tokens", 0)
            for result in results
        )
        avg_tokens = total_tokens / len(results) if results else 0

        # Compile final report
        eval_report = {
            "status": "success",
            "num_questions": len(results),
            "avg_score": avg_score,
            "intent_accuracy": intent_accuracy,
            "topic_accuracy": topic_accuracy,
            "content_accuracy": content_accuracy,
            "total_tokens": total_tokens,
            "avg_tokens_per_query": avg_tokens,
            "results": results,
        }

        # Save results
        with open(output_path, "w") as f:
            json.dump(eval_report, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {output_path}")
        logger.info(f"Average score: {avg_score:.2f}")
        logger.info(f"Intent accuracy: {intent_accuracy:.2f}")
        logger.info(f"Topic accuracy: {topic_accuracy:.2f}")
        logger.info(f"Content accuracy: {content_accuracy:.2f}")

        return eval_report

    def generate_report(
        self, eval_results: Dict[str, Any], output_path: Optional[Path] = None
    ) -> Path:
        """Generate a human-readable HTML report from evaluation results."""
        output_path = output_path or self.eval_dir / "eval_report.html"

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strathmore RAG Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #003366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .score-high {{ color: green; }}
                .score-medium {{ color: orange; }}
                .score-low {{ color: red; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-card {{ background-color: #f2f2f2; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
            </style>
        </head>
        <body>
            <h1>Strathmore RAG Evaluation Report</h1>

            <h2>Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Average Score</h3>
                    <p class="{self._get_score_class(eval_results.get('avg_score', 0))}">{eval_results.get('avg_score', 0):.2f}</p>
                </div>
                <div class="metric-card">
                    <h3>Intent Accuracy</h3>
                    <p class="{self._get_score_class(eval_results.get('intent_accuracy', 0))}">{eval_results.get('intent_accuracy', 0):.2f}</p>
                </div>
                <div class="metric-card">
                    <h3>Topic Accuracy</h3>
                    <p class="{self._get_score_class(eval_results.get('topic_accuracy', 0))}">{eval_results.get('topic_accuracy', 0):.2f}</p>
                </div>
                <div class="metric-card">
                    <h3>Content Accuracy</h3>
                    <p class="{self._get_score_class(eval_results.get('content_accuracy', 0))}">{eval_results.get('content_accuracy', 0):.2f}</p>
                </div>
            </div>

            <h2>Token Usage</h2>
            <p>Total Tokens: {eval_results.get('total_tokens', 0)}</p>
            <p>Average Tokens Per Query: {eval_results.get('avg_tokens_per_query', 0):.2f}</p>

            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Query</th>
                    <th>Score</th>
                    <th>Intent (Expected/Actual)</th>
                    <th>Topic (Expected/Actual)</th>
                    <th>Contains Expected</th>
                </tr>
        """

        # Add rows for each result
        for result in eval_results.get("results", []):
            row = f"""
                <tr>
                    <td>{result.get('id', '')}</td>
                    <td>{result.get('query', '')}</td>
                    <td class="{self._get_score_class(result.get('score', 0))}">{result.get('score', 0):.2f}</td>
                    <td>
                        {result.get('expected_intent', '')}/{result.get('actual_intent', '')}
                        {'✓' if result.get('intent_match', False) else '✗'}
                    </td>
                    <td>
                        {result.get('expected_topic', '')}/{result.get('actual_topic', '')}
                        {'✓' if result.get('topic_match', False) else '✗'}
                    </td>
                    <td>{'✓' if result.get('contains_expected', False) else '✗'}</td>
                </tr>
            """
            html_content += row

        # Close HTML
        html_content += """
            </table>
        </body>
        </html>
        """

        # Save HTML report
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report generated at {output_path}")
        return output_path

    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score value."""
        if score >= 0.7:
            return "score-high"
        elif score >= 0.4:
            return "score-medium"
        else:
            return "score-low"
