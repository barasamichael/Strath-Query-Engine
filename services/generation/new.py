import os
import logging
from typing import Any
from typing import Dict
from typing import List

import openai
from openai import OpenAI

from config.settings import settings
from services.generation.intent_recognizer import IntentType

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

        # Build prompt
        messages = [
            {"role": "system", "content": system_instruction},
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
You are an assistant for Strathmore University students, providing EXHAUSTIVE and COMPREHENSIVE information from the student handbook. 
Always ground your responses in the provided context.
Your goal is to provide the MOST COMPLETE information possible from the provided chunks.
Look through ALL the chunks carefully to find ANY relevant information for the query.
Even if information is scattered across multiple chunks, combine it to create a complete response.
If multiple chunks contain overlapping or related information, synthesize them to provide all details.
Structure your responses with clear organization, using bullet points and headings when appropriate.
Never make up information or hallucinate facts that aren't in the context.
When information is truly missing, clearly state that it's not available in the handbook.
"""

        intent_specific_instructions = {
            IntentType.FACTUAL_QUERY: """
THOROUGHLY EXAMINE ALL CONTEXT CHUNKS for information related to the query.
Focus on providing precise, factual information from the handbook. Be exhaustive and comprehensive.
Include ALL relevant details from the context, combining information from multiple chunks when they relate to the same topic.
Organize information logically with proper structure.
Be especially thorough when searching for policy information, requirements, or specific rules.
""",
            IntentType.PROCEDURAL_QUERY: """
SEARCH EXHAUSTIVELY through all context chunks for procedural information.
Provide step-by-step instructions or explain processes clearly and thoroughly.
Organize information in a logical sequence and highlight important deadlines or requirements.
Include ALL relevant details from different chunks to ensure the instructions are complete.
Use numbered lists for procedures and bullet points for requirements when appropriate.
""",
            IntentType.EXPLANATION_QUERY: """
EXAMINE ALL CONTEXT CHUNKS thoroughly for explanatory information.
Explain concepts thoroughly with examples when available.
Clarify underlying reasons or principles that inform policies or procedures.
Synthesize information from multiple chunks to provide a complete understanding.
Use a structured approach, breaking down complex topics into digestible parts.
""",
            IntentType.COMPARISON_QUERY: """
SEARCH ALL CONTEXT CHUNKS comprehensively for comparison information.
Highlight similarities and differences between the compared items exhaustively.
Use structured formatting to make comparisons clear and easy to understand.
Pull in all relevant information from different chunks to ensure the comparison is complete.
Organize information in tables or parallel structures when appropriate.
""",
            IntentType.OFF_TOPIC: """
Before concluding a topic is off-topic, CAREFULLY CHECK ALL CHUNKS for any relevant information.
Even if the query seems unrelated, look for indirect connections or partial information.
If truly outside scope, politely inform the user that their question appears to be outside the scope of the Strathmore University handbook.
Suggest that they might want to ask about topics related to the university instead.
Do not attempt to answer off-topic questions with made-up information.
""",
            IntentType.CLARIFICATION: """
THOROUGHLY EXAMINE ALL CONTEXT CHUNKS for clarification information.
Address the specific point the user is asking for clarification about.
Provide additional context or examples to enhance understanding.
Pull together information from multiple chunks for a more complete explanation.
Check if there are multiple interpretations of the original statement and address them if needed.
""",
            IntentType.FEEDBACK: """
SEARCH ALL CONTEXT CHUNKS for relevant feedback information.
Acknowledge the feedback professionally.
If there's a question within the feedback, focus on answering that question thoroughly.
Pull information from all relevant chunks to provide a comprehensive response.
""",
            IntentType.GENERAL_CHAT: """
EXAMINE ALL CONTEXT CHUNKS thoroughly before responding.
Keep the tone conversational but professional.
If appropriate, gently steer the conversation toward providing helpful information about Strathmore University.
Look for relevant information across all context chunks that might be helpful.
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
1. EXAMINE ALL CONTEXT CHUNKS EXHAUSTIVELY to ensure you don't miss any relevant information.
2. Provide COMPREHENSIVE information by combining details from all relevant context chunks.
3. Include clear citations for all factual statements when possible.
4. Structure your response with appropriate headings, bullet points, or numbering for clarity.
5. If information is missing, acknowledge the limitation and suggest alternatives.
6. Use a helpful, concise, and educational tone appropriate for university students.
7. Always prioritize COMPLETENESS over brevity - include ALL relevant information from the chunks.
8. When the handbook contains partial information, provide what's available rather than saying nothing is available.
"""

        intent_specific_guidelines = {
            IntentType.FACTUAL_QUERY: """
9. Present facts directly and clearly, organizing related information together.
10. Include ALL relevant details from the context, not just the most obvious ones.
11. When multiple chunks contain related information, synthesize them into a coherent whole.
12. Be especially thorough when providing information about policies, requirements, or regulations.
""",
            IntentType.PROCEDURAL_QUERY: """
9. Present steps in a clear, numbered sequence.
10. Include ALL aspects of the procedure from different chunks to ensure completeness.
11. Highlight important deadlines, requirements, or potential obstacles.
12. If the complete procedure isn't available, mention what parts are known and what might be missing.
""",
            IntentType.EXPLANATION_QUERY: """
9. Explain concepts thoroughly, building from basic principles.
10. Combine information from multiple chunks to create a complete explanation.
11. Use examples to illustrate points when possible.
12. Distinguish between facts, policies, and interpretations.
""",
            IntentType.COMPARISON_QUERY: """
9. Clearly identify the items being compared.
10. Pull information from all relevant chunks to ensure a comprehensive comparison.
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
9. Focus specifically on the point being clarified.
10. Provide additional context or examples from different chunks to enhance understanding.
11. Synthesize information from multiple sources to create a more complete picture.
12. Check if there are multiple interpretations of the original statement and address them if needed.
""",
            IntentType.FEEDBACK: """
9. Acknowledge the feedback professionally.
10. If there's a question within the feedback, focus on answering that question comprehensively.
11. Pull together all relevant information from different chunks to provide a thorough response.
""",
            IntentType.GENERAL_CHAT: """
9. Keep the tone conversational but professional.
10. Look for any relevant information across all context chunks that might be helpful.
11. If appropriate, gently steer the conversation toward providing helpful information about Strathmore University.
12. Provide information even if it's only tangentially related to the query, as long as it's from the handbook.
""",
        }

        return common_guidelines + intent_specific_guidelines.get(
            intent_type, ""
        )
