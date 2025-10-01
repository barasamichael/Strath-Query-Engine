import os
import logging
from typing import Any
from typing import Dict
from typing import List

import openai
from openai import OpenAI

from config.settings import settings
from services.intent_recognizer import IntentType

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_generator")

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ResponseGenerator:
    def __init__(self):
        self.model = settings.llm.model
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

    def generate_response(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        intent_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a response to the user query based on the retrieved context and detected intent.
        """
        # Prepare system instruction based on intent
        system_instruction = self._get_system_instruction(
            intent_info["intent_type"]
        )

        # Format retrieved context
        context_text = self._format_context(retrieved_context)

        # Prepare response guidelines based on intent
        response_guidelines = self._get_response_guidelines(
            intent_info["intent_type"]
        )

        # Add implicit Strathmore context
        implicit_context = """
You are providing information about Strathmore University based on the student handbook.
All questions should be interpreted in the context of Strathmore University unless they are clearly about something else.
If a question lacks specific context, assume it's about Strathmore University.
"""

        # Build prompt
        messages = [
            {"role": "system", "content": system_instruction + implicit_context},
            {
                "role": "user",
                "content": f"""
[CONTEXT]
{context_text}

[USER INTENT]
The user's question relates to {intent_info["intent_type"].value} and appears to be asking about {intent_info["topic"].value}.

[QUERY]
{query}

[RESPONSE GUIDELINES]
{response_guidelines}
""",
            },
        ]

        # Call the LLM
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract response content
            response_content = response.choices[0].message.content

            # For debugging - calculate token usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            logger.info(
                f"Generated response with {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {total_tokens} total tokens"
            )

            return {
                "response": response_content,
                "intent_type": intent_info["intent_type"],
                "topic": intent_info["topic"],
                "confidence": intent_info["confidence"],
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try again or contact support if the issue persists.",
                "intent_type": intent_info["intent_type"],
                "topic": intent_info["topic"],
                "confidence": intent_info["confidence"],
                "error": str(e),
            }

    def _get_system_instruction(self, intent_type: IntentType) -> str:
        """Get the system instruction based on the intent type."""
        base_instruction = """
You are an assistant for Strathmore University students, providing PRECISE and FOCUSED information from the student handbook.
Ground your responses in the provided context.
Your goal is to provide RELEVANT information directly addressing the user's query.
Look through the chunks to find specifically relevant information for the query.
Synthesize information from multiple chunks when necessary, but focus on answering the specific question.
Structure your responses clearly, using bullet points and headings when appropriate.
Never make up information or hallucinate facts that aren't in the context.
When information is truly missing, clearly state that it's not available in the handbook.
Avoid including tangential or loosely related information that doesn't directly answer the query.
"""

        intent_specific_instructions = {
            IntentType.FACTUAL_QUERY: """
EXAMINE CONTEXT CHUNKS for information DIRECTLY related to the query.
Focus on providing precise, factual information specifically addressing the question.
Include relevant details that directly answer the query, avoiding tangential information.
Organize information logically with proper structure.
Be especially focused when providing policy information, requirements, or specific rules.
""",
            IntentType.PROCEDURAL_QUERY: """
SEARCH context chunks for procedural information directly relevant to the query.
Provide clear step-by-step instructions or explain processes directly related to the question.
Organize information in a logical sequence and highlight important deadlines or requirements.
Include only the details necessary to complete the process or understand the procedure.
Use numbered lists for procedures and bullet points for requirements when appropriate.
""",
            IntentType.EXPLANATION_QUERY: """
EXAMINE context chunks for explanatory information directly addressing the query.
Explain concepts clearly and concisely, focusing on what the user is specifically asking about.
Clarify underlying reasons or principles that are directly relevant to the question.
Synthesize information from multiple chunks if needed, but maintain focus on the specific question.
Use a structured approach, breaking down complex topics into digestible parts.
""",
            IntentType.COMPARISON_QUERY: """
SEARCH context chunks for comparison information directly relevant to the query.
Focus on the specific items being compared in the user's question.
Use structured formatting to make comparisons clear and easy to understand.
Pull in only the information from chunks that directly relates to the comparison requested.
Organize information in tables or parallel structures when appropriate.
""",
            IntentType.OFF_TOPIC: """
Before concluding a topic is off-topic, check chunks for any relevant information.
If truly outside scope, politely inform the user that their question appears to be outside the scope of the Strathmore University handbook.
Suggest that they might want to ask about topics related to the university instead.
Do not attempt to answer off-topic questions with made-up information.
""",
            IntentType.CLARIFICATION: """
Address the specific point the user is asking for clarification about.
Provide additional context or examples that directly clarify the point in question.
Pull together information from multiple chunks if needed, but maintain focus.
Be concise and direct in your clarification.
""",
            IntentType.FEEDBACK: """
Acknowledge the feedback professionally.
If there's a question within the feedback, focus on answering that question directly.
Keep responses concise and relevant to any questions asked.
""",
            IntentType.GENERAL_CHAT: """
Keep the tone conversational but professional.
Provide relevant information about Strathmore University that might be helpful.
Be concise and avoid unnecessary details unless specifically requested.
""",
        }

        return base_instruction + intent_specific_instructions.get(
            intent_type, ""
        )

    def _format_context(self, retrieved_context: List[Dict[str, Any]]) -> str:
        """Format retrieved context chunks for inclusion in the prompt."""
        if not retrieved_context:
            return "No relevant information found in the handbook."

        formatted_chunks = []
        for i, chunk in enumerate(retrieved_context):
            formatted_chunk = f"[CHUNK {i+1}] (Relevance: {chunk['score']:.2f})\n{chunk['text']}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n".join(formatted_chunks)

    def _get_response_guidelines(self, intent_type: IntentType) -> str:
        """Get response guidelines based on the intent type."""
        common_guidelines = """
1. Focus on information DIRECTLY RELEVANT to the user's query.
2. Combine details from relevant context chunks that specifically address the question.
3. Include clear citations for factual statements when possible.
4. Structure your response with appropriate headings, bullet points, or numbering for clarity.
5. If information is missing, acknowledge the limitation and suggest alternatives.
6. Use a helpful, concise, and educational tone appropriate for university students.
7. Prioritize RELEVANCE over completeness - include only information that directly answers the query.
8. When the handbook contains partial information, provide what's available rather than saying nothing is available.
"""

        intent_specific_guidelines = {
            IntentType.FACTUAL_QUERY: """
9. Present facts directly and clearly, organizing related information together.
10. Focus on the specific facts requested and avoid tangential information.
11. When multiple chunks contain related information, synthesize them into a focused answer.
12. Be concise when providing information about policies, requirements, or regulations.
""",
            IntentType.PROCEDURAL_QUERY: """
9. Present steps in a clear, numbered sequence.
10. Include only the aspects of the procedure that are directly relevant to the query.
11. Highlight important deadlines, requirements, or potential obstacles.
12. If the complete procedure isn't available, mention what parts are known and what might be missing.
""",
            IntentType.EXPLANATION_QUERY: """
9. Explain concepts clearly, focusing on what the user specifically asked about.
10. Combine only the most relevant information from chunks to create a focused explanation.
11. Use examples to illustrate points when helpful and directly relevant.
12. Distinguish between facts, policies, and interpretations.
""",
            IntentType.COMPARISON_QUERY: """
9. Clearly identify the items being compared and focus only on those items.
10. Pull only the most relevant information from chunks for the comparison.
11. Highlight key similarities and differences in a structured way.
12. Avoid making subjective judgments about which option is "better" unless explicitly stated in the context.
""",
            IntentType.OFF_TOPIC: """
9. Before concluding a topic is off-topic, carefully check all chunks for any relevant information.
10. If truly outside scope, politely state that the question appears to be outside the scope of information about Strathmore University.
11. Do not attempt to answer with made-up information.
12. Suggest that the user might want to ask about topics related to Strathmore University instead.
""",
            IntentType.CLARIFICATION: """
9. Focus specifically on the point being clarified without adding unnecessary information.
10. Provide additional context or examples only if they directly help clarify the point in question.
11. Be concise and direct in your clarification.
12. If there are multiple interpretations, focus on the most likely one based on context.
""",
            IntentType.FEEDBACK: """
9. Acknowledge the feedback professionally and concisely.
10. If there's a question within the feedback, focus on answering that question directly.
11. Keep responses brief and to the point.
""",
            IntentType.GENERAL_CHAT: """
9. Keep the tone conversational but professional.
10. Provide only information that might be directly helpful to the user's query.
11. Be concise and avoid lengthy explanations unless specifically requested.
12. If appropriate, suggest specific topics the user might want to learn more about.
""",
        }

        return common_guidelines + intent_specific_guidelines.get(
            intent_type, ""
        )
