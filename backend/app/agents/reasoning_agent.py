"""
Reasoning Agent for question answering with LLM.

This agent takes retrieved context and uses an LLM to generate an answer to the user's
question. It implements strict grounding - the LLM is instructed to ONLY use information
from the provided context, reducing hallucination risk.

Agent Pattern:
This implements a simple reasoning agent that:
1. Receives a question and context
2. Constructs a grounded prompt
3. Calls the LLM for an answer
4. Validates the response

No complex chain-of-thought or multi-step reasoning - focused on accuracy and grounding.
"""

import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
# Check several possible locations for .env
env_locations = [
    Path('.') / '.env',
    Path(__file__).parent.parent.parent / '.env',
    Path.cwd() / '.env',
    Path.cwd() / 'backend' / '.env'
]

for loc in env_locations:
    if loc.exists():
        load_dotenv(dotenv_path=loc, override=True)
        if os.getenv("GROQ_API_KEY"):
            logger.info(f"Loaded GROQ_API_KEY from {loc.absolute()}")
            break

from app.agents.retrieval_agent import RetrievedContext
from app.utils.logger import get_agent_logger, log_event

logger = get_agent_logger()


class ReasoningAgent:
    """
    Agent responsible for answering questions using LLM with retrieved context.
    
    Key Features:
    - Strict grounding: Only answers based on provided context
    - multi-provider logic: Supports Groq (primary) and OpenAI (secondary)
    - Graceful fallback: Uses mock responses if no API is available
    - Prompt loading: Loads QA prompt from file for easy customization
    """
    
    def __init__(self, model: str = "llama3-8b-8192", temperature: float = 0.0):
        """
        Initialize the reasoning agent.
        
        Args:
            model: Primary model to use (Groq model)
            temperature: LLM temperature (0 = deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        self.client = None
        self.provider = None

        print(f"DEBUG: Initializing ReasoningAgent. GROQ_KEY present: {bool(self.groq_api_key)}")

        # Try Groq first
        if self.groq_api_key:
            try:
                from langchain_groq import ChatGroq
                self.client = ChatGroq(
                    api_key=self.groq_api_key,
                    model_name=self.model,
                    temperature=self.temperature
                )
                self.provider = "groq"
                logger.info(f"ReasoningAgent successfully initialized with Groq model={self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")
                logger.warning("Falling back to OpenAI...")
        
        # Fallback to OpenAI
        if not self.provider and self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.model = "gpt-3.5-turbo"
                self.provider = "openai"
                logger.info(f"ReasoningAgent initialized with OpenAI model={self.model}")
            except ImportError:
                logger.warning("OpenAI package not installed.")

        self.use_mock = self.provider is None
        
        if self.use_mock:
            logger.warning("No valid API Key (GROQ or OPENAI) found. Using mock LLM responses.")
        
        # Load the QA prompt template
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the QA prompt template from file."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "qa_prompt.txt"
        
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found at {prompt_path}. Using default.")
            return self._default_prompt_template()
    
    def _default_prompt_template(self) -> str:
        """Return the default prompt template."""
        return """You are a precise question-answering assistant.

Answer ONLY using the provided context below.
If the information needed to answer is not present in the context, respond with:
"Information not found in documents."

Do NOT make up information or use external knowledge.
Be concise and factual in your response.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    def reason(
        self,
        question: str,
        contexts: List[RetrievedContext]
    ) -> str:
        """
        Generate an answer to the question using retrieved context.
        
        Args:
            question: The user's question
            contexts: List of retrieved context chunks
        
        Returns:
            The generated answer string
        """
        log_event(logger, "REASONING_STARTED", {
            "question_length": len(question),
            "context_chunks": len(contexts)
        })
        
        # Format context for the prompt
        context_text = self._format_context(contexts)
        
        # Build the prompt
        prompt = self.prompt_template.format(
            context=context_text,
            question=question
        )
        
        # Generate answer
        try:
            if self.use_mock:
                answer = self._mock_response(question, contexts)
            else:
                answer = self._call_llm(prompt)
                
                # If LLM call returned an error message string (legacy behavior we want to avoid), 
                # or if we want to be safe, we check if it looks like an error
                if answer.startswith("Error generating answer:"):
                    logger.warning(f"LLM call returned error. Falling back to mock. Error: {answer}")
                    answer = self._mock_response(question, contexts)
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}. Falling back to mock.")
            answer = self._mock_response(question, contexts)
        
        log_event(logger, "REASONING_COMPLETED", {
            "answer_length": len(answer)
        })
        
        return answer
    
    def _format_context(self, contexts: List[RetrievedContext]) -> str:
        """Format context chunks for the prompt."""
        if not contexts:
            return "No relevant context was found in the documents."
        
        formatted_parts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_parts.append(
                f"[Source {i}]\n{ctx.content}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM provider to generate an answer.
        
        Args:
            prompt: The formatted prompt
        
        Returns:
            The LLM's response
        """
        try:
            log_event(logger, "LLM_CALL_STARTED", {"provider": self.provider, "model": self.model})
            
            if self.provider == "groq":
                response = self.client.invoke([
                    {"role": "system", "content": "You are a precise assistant that only answers based on provided context."},
                    {"role": "user", "content": prompt}
                ])
                return response.content
            
            elif self.provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise assistant that only answers based on provided context."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content.strip()
                
                log_event(logger, "LLM_CALL_COMPLETED", {
                    "tokens_used": response.usage.total_tokens if response.usage else "N/A"
                })
                
                return answer
            
            return "No valid provider configured."
            
        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}): {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _mock_response(
        self,
        question: str,
        contexts: List[RetrievedContext]
    ) -> str:
        """
        Generate a mock response for testing without OpenAI/Groq API.
        """
        if not contexts:
            return "Information not found in documents."
        
        # Create a mock answer based on available context
        context_preview = contexts[0].content[:400] if contexts else ""
        
        return (
            f"### [System Notice: AI Model Offline]\n\n"
            f"**Action Required**: To enable live AI answers, please follow these steps:\n"
            f"1. Open the file `backend/.env` in your editor.\n"
            f"2. Add your key: `GROQ_API_KEY=gsk_your_key_here`\n"
            f"3. Restart the system.\n\n"
            f"---\n\n"
            f"**Document Content Found**:\n"
            f"> \"...{context_preview}...\"\n\n"
            f"**Summary**: The system successfully retrieved {len(contexts)} relevant chunks from your document. "
            f"Once you configure the Groq API key, the Llama3 model will process this information into a natural answer."
        )
    
    def validate_grounding(self, answer: str, contexts: List[RetrievedContext]) -> bool:
        """
        Validate that the answer is grounded in the provided context.
        
        This is a simple validation that checks if key terms from the answer
        appear in the context. A production system might use more sophisticated
        NLI (Natural Language Inference) models.
        
        Args:
            answer: The generated answer
            contexts: The context chunks used
        
        Returns:
            True if answer appears to be grounded
        """
        if "Information not found" in answer:
            return True  # Valid response for missing information
        
        # Simple word overlap check
        context_text = " ".join(ctx.content.lower() for ctx in contexts)
        answer_words = set(answer.lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        answer_words = answer_words - stop_words
        
        # Check if significant words from answer appear in context
        if not answer_words:
            return True
        
        overlap = sum(1 for word in answer_words if word in context_text)
        overlap_ratio = overlap / len(answer_words)
        
        # Consider grounded if >30% of answer words appear in context
        is_grounded = overlap_ratio > 0.3
        
        log_event(logger, "GROUNDING_CHECK", {
            "overlap_ratio": f"{overlap_ratio:.2f}",
            "is_grounded": is_grounded
        })
        
        return is_grounded
